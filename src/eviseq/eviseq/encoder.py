"""Native Qwen dual-mask encoder and native embedding-model control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .native_attention import evidence_key_attention_bias, mix_attention_outputs, pool_units, sdpa_mask


@dataclass
class EncoderOutput:
    memory: torch.Tensor
    unit_logits: Optional[torch.Tensor]
    valid_units: Optional[torch.Tensor]
    native_gate_mean: torch.Tensor


def _native_core(model: nn.Module) -> nn.Module:
    candidate = getattr(model, "model", model)
    required = ("embed_tokens", "layers", "norm", "rotary_emb")
    if not all(hasattr(candidate, name) for name in required):
        raise TypeError(
            "qwen_native requires an AutoModel exposing embed_tokens/layers/norm/rotary_emb; "
            f"got {type(model).__name__}"
        )
    return candidate


class NativeDualMaskQwenEncoder(nn.Module):
    """Reuse every pretrained Qwen parameter while changing only attention flow."""

    def __init__(self, config: Dict[str, Any], dtype: torch.dtype):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        model_config = config["model"]
        attention_config = config["native_attention"]
        name = str(model_config["encoder_name"])
        trust_remote_code = bool(model_config.get("encoder_trust_remote_code", True))
        revision = model_config.get("encoder_revision")
        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if revision:
            kwargs["revision"] = str(revision)
        pretrained = AutoConfig.from_pretrained(name, **kwargs)
        if str(getattr(pretrained, "model_type", "")) != "qwen3":
            raise ValueError(f"qwen_native currently supports Qwen3 checkpoints, not {pretrained.model_type!r}")
        implementation = str(model_config.get("encoder_attn_implementation", "sdpa"))
        self.model = AutoModel.from_pretrained(
            name,
            config=pretrained,
            dtype=dtype,
            attn_implementation=implementation,
            **kwargs,
        )
        self.config = pretrained
        self.model_name = name
        self.hidden_size = int(pretrained.hidden_size)
        self.num_hidden_layers = int(pretrained.num_hidden_layers)
        self.num_heads = int(pretrained.num_attention_heads)
        self.variant = str(attention_config.get("variant", "evidence"))
        self.evidence_key_bias_scale = float(attention_config.get("evidence_key_bias_scale", 1.0))
        self.attn_implementation = implementation
        self.gradient_checkpointing = bool(model_config.get("gradient_checkpointing", True))
        layer_types = list(getattr(pretrained, "layer_types", ["full_attention"] * self.num_hidden_layers))
        if any(value != "full_attention" for value in layer_types):
            raise ValueError("EviSeq fails closed on sliding-attention Qwen variants")

        salience_hidden = int(config["bridge"].get("salience_hidden_size", max(128, self.hidden_size // 2)))
        dropout = float(config["bridge"].get("dropout", 0.0))
        self.evidence_norm = nn.RMSNorm(self.hidden_size)
        self.evidence_head = nn.Sequential(
            nn.Linear(self.hidden_size, salience_hidden, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(salience_hidden, 1, bias=True),
        )
        nn.init.zeros_(self.evidence_head[-1].weight)
        nn.init.zeros_(self.evidence_head[-1].bias)
        self.evidence_view_gate = nn.Parameter(torch.zeros(self.num_hidden_layers, self.num_heads))
        self.generic_token_gate = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        nn.init.zeros_(self.generic_token_gate.weight)
        nn.init.zeros_(self.generic_token_gate.bias)
        self.model.config.use_cache = False

    @property
    def core(self) -> nn.Module:
        return _native_core(self.model)

    def _unit_logits(
        self,
        hidden_states: torch.Tensor,
        unit_ids: Optional[torch.Tensor],
        unit_count: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if unit_ids is None:
            return None, None
        if unit_count is None:
            unit_count = int(unit_ids.max().item())
        units, valid = pool_units(self.evidence_norm(hidden_states), unit_ids, unit_count)
        return self.evidence_head(units).squeeze(-1), valid

    def _attention(
        self,
        attention: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        layer_index: int,
        unit_ids: Optional[torch.Tensor],
        unit_count: Optional[int],
    ) -> torch.Tensor:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query = attention.q_norm(attention.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.attn_implementation, None)
        if interface is None:
            raise RuntimeError(f"Transformers has no attention interface {self.attn_implementation!r}")

        def attend(is_causal: bool, backend_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            if self.attn_implementation == "flash_attention_2":
                if backend_mask is not None:
                    raise RuntimeError("Evidence-key routing requires the audited SDPA attention backend")
                backend_mask = attention_mask if not bool(attention_mask.bool().all()) else None
            elif backend_mask is None:
                backend_mask = sdpa_mask(attention_mask, is_causal, hidden_states.shape[1])
            output, _ = interface(
                attention,
                query,
                key,
                value,
                backend_mask,
                dropout=0.0 if not self.training else float(attention.attention_dropout),
                scaling=attention.scaling,
                sliding_window=None,
                is_causal=is_causal,
            )
            return output

        if self.variant == "causal":
            mixed = attend(True)
        elif self.variant == "full":
            mixed = attend(False)
        else:
            causal = attend(True)
            full = None
            generic_logits = None
            if self.variant == "evidence":
                logits, valid = self._unit_logits(hidden_states, unit_ids, unit_count)
                if logits is None or valid is None or unit_ids is None:
                    raise ValueError("evidence attention requires source unit_ids")
                key_bias = evidence_key_attention_bias(
                    logits,
                    valid,
                    unit_ids,
                    attention_mask,
                    dtype=query.dtype,
                    scale=self.evidence_key_bias_scale,
                )
                full = attend(False, key_bias)
            elif self.variant == "dec2enc":
                full = attend(False)
                generic_logits = self.generic_token_gate(self.evidence_norm(hidden_states))
            if full is None:
                raise RuntimeError(f"Dual-mask variant {self.variant!r} did not construct a future view")
            mixed = mix_attention_outputs(
                causal,
                full,
                self.variant,
                self.evidence_view_gate[layer_index],
                generic_logits=generic_logits,
            )
        output = mixed.reshape(*input_shape, -1).contiguous()
        output = attention.o_proj(output)
        return output.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)

    def _layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        unit_ids: Optional[torch.Tensor],
        unit_count: Optional[int],
        layer_index: int,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        hidden_states = residual + self._attention(
            layer.self_attn,
            normalized,
            position_embeddings,
            attention_mask,
            layer_index,
            unit_ids,
            unit_count,
        )
        residual = hidden_states
        hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))
        return hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
    ) -> EncoderOutput:
        hidden_states = self.core.embed_tokens(input_ids)
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(~attention_mask.bool(), 0)
        position_embeddings = self.core.rotary_emb(hidden_states, position_ids)
        unit_count = int(unit_ids.max().item()) if unit_ids is not None else None
        for index, layer in enumerate(self.core.layers[: self.num_hidden_layers]):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():

                def custom_forward(states: torch.Tensor, *, _layer=layer, _index=index) -> torch.Tensor:
                    return self._layer(
                        _layer,
                        states,
                        position_embeddings,
                        attention_mask,
                        unit_ids,
                        unit_count,
                        _index,
                    )

                hidden_states = torch.utils.checkpoint.checkpoint(custom_forward, hidden_states, use_reentrant=False)
            else:
                hidden_states = self._layer(
                    layer,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    unit_ids,
                    unit_count,
                    index,
                )
        memory = self.core.norm(hidden_states)
        memory = memory.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
        logits, valid = self._unit_logits(memory, unit_ids, unit_count)
        gate = torch.tanh(self.evidence_view_gate.float()).abs().mean()
        return EncoderOutput(memory, logits, valid, gate)

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = bool(trainable)
        for module in (self.evidence_norm, self.evidence_head, self.generic_token_gate):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.evidence_view_gate.requires_grad = True


class PretrainedNativeEncoder(nn.Module):
    """Token-memory path for PPLX/Nemotron comparison in the same final graph."""

    def __init__(self, config: Dict[str, Any], dtype: torch.dtype):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        model_config = config["model"]
        name = str(model_config["encoder_name"])
        kwargs: Dict[str, Any] = {"trust_remote_code": bool(model_config.get("encoder_trust_remote_code", True))}
        revision = model_config.get("encoder_revision")
        if revision:
            kwargs["revision"] = str(revision)
        pretrained = AutoConfig.from_pretrained(name, **kwargs)
        self.model = AutoModel.from_pretrained(
            name,
            config=pretrained,
            dtype=dtype,
            attn_implementation=str(model_config.get("encoder_attn_implementation", "sdpa")),
            **kwargs,
        )
        self.config = pretrained
        self.model_name = name
        self.hidden_size = int(pretrained.hidden_size)
        self.gradient_checkpointing = bool(model_config.get("gradient_checkpointing", True))
        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        salience_hidden = int(config["bridge"].get("salience_hidden_size", max(128, self.hidden_size // 2)))
        self.evidence_norm = nn.RMSNorm(self.hidden_size)
        self.evidence_head = nn.Sequential(
            nn.Linear(self.hidden_size, salience_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(salience_hidden, 1, bias=True),
        )
        nn.init.zeros_(self.evidence_head[-1].weight)
        nn.init.zeros_(self.evidence_head[-1].bias)
        self.model.config.use_cache = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
    ) -> EncoderOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        memory = outputs.last_hidden_state.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
        if unit_ids is None:
            logits, valid = None, None
        else:
            units, valid = pool_units(self.evidence_norm(memory), unit_ids, int(unit_ids.max().item()))
            logits = self.evidence_head(units).squeeze(-1)
        return EncoderOutput(memory, logits, valid, memory.float().new_zeros(()))

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = bool(trainable)
        for module in (self.evidence_norm, self.evidence_head):
            for parameter in module.parameters():
                parameter.requires_grad = True


def build_encoder(config: Dict[str, Any], dtype: torch.dtype) -> nn.Module:
    backend = str(config["native_attention"].get("backend", "qwen_native"))
    if backend == "qwen_native":
        return NativeDualMaskQwenEncoder(config, dtype)
    return PretrainedNativeEncoder(config, dtype)
