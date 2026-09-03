"""Full pretrained Qwen3 decoder with conventional copied cross-attention."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import align_trainable_sdpa_bias_heads, ensure_sdpa_lse_for_bias_backward

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    GradientCheckpointingLayer = nn.Module


def _load_causal_lm(model_name: str, dtype: torch.dtype) -> Tuple[Any, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM

    raw_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config = raw_config.get_text_config() if hasattr(raw_config, "get_text_config") else raw_config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    return model, config


class QwenCopiedCrossAttention(nn.Module):
    """Independent source attention initialized from the same decoder layer.

    Unlike merged attention, source keys receive their own softmax exactly as
    in the pretrained decoder's coordinate system. Q/K/V/O begin from the
    corresponding self-attention weights when requested.
    """

    def __init__(
        self,
        source_attention: nn.Module,
        input_norm: nn.Module,
        config: Any,
        dropout: float,
        initialize_from_self: bool,
    ):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.num_heads))
        self.q_proj = copy.deepcopy(source_attention.q_proj)
        self.k_proj = copy.deepcopy(source_attention.k_proj)
        self.v_proj = copy.deepcopy(source_attention.v_proj)
        self.o_proj = copy.deepcopy(source_attention.o_proj)
        self.q_norm = copy.deepcopy(getattr(source_attention, "q_norm", nn.Identity()))
        self.k_norm = copy.deepcopy(getattr(source_attention, "k_norm", nn.Identity()))
        self.memory_norm = copy.deepcopy(input_norm)
        self.dropout = float(dropout)
        self._memory_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._sdpa_gqa_supported: Optional[bool] = None
        if not initialize_from_self:
            initializer_range = float(getattr(config, "initializer_range", 0.02))
            for module in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            for norm in (self.q_norm, self.k_norm):
                if hasattr(norm, "weight"):
                    nn.init.ones_(norm.weight)

    def _project_memory(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, source_length, _ = memory.shape
        normalized = self.memory_norm(memory)
        key = self.k_proj(normalized).view(batch, source_length, self.num_kv_heads, self.head_dim)
        value = self.v_proj(normalized).view(batch, source_length, self.num_kv_heads, self.head_dim)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        return key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        encoder_attention_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, query_length, self.num_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2).contiguous()
        cached = self._memory_cache if not self.training else None
        if cached is None:
            if encoder_hidden_states.ndim != 3:
                raise ValueError("Cross-attention requires one routed [B, S, D] memory")
            key, value = self._project_memory(encoder_hidden_states)
        else:
            key, value = cached
            if (
                key.shape[0] != batch
                or key.shape[-2] != encoder_hidden_states.shape[1]
                or key.device != query.device
                or value.device != query.device
                or key.dtype != query.dtype
                or value.dtype != query.dtype
            ):
                raise RuntimeError("Stale cross-attention cache; rebuild it for the current source batch")
        source_length = key.shape[-2]
        if value.shape[-2] != source_length or key.shape[-1] != self.head_dim or value.shape[-1] != self.head_dim:
            raise RuntimeError("Cross-attention K/V cache has an incompatible shape")
        if encoder_attention_mask is not None and tuple(encoder_attention_mask.shape) != (batch, source_length):
            raise ValueError(
                "encoder_attention_mask must have shape [batch, source_length] matching cross-attention memory"
            )
        mask = None
        if encoder_attention_bias is not None:
            if encoder_attention_bias.ndim == 2:
                if tuple(encoder_attention_bias.shape) != (batch, source_length):
                    raise ValueError(
                        "encoder_attention_bias must have shape [batch, source_length] matching cross-attention memory"
                    )
                mask = encoder_attention_bias.to(query.dtype)[:, None, None, :]
            elif encoder_attention_bias.ndim == 4:
                if (
                    tuple(encoder_attention_bias.shape[-2:]) != (1, source_length)
                    or encoder_attention_bias.shape[0] != batch
                    or encoder_attention_bias.shape[1] != 1
                ):
                    raise ValueError("4-D encoder_attention_bias must have shape [batch, 1, 1, source_length]")
                mask = encoder_attention_bias.to(query.dtype)
            else:
                raise ValueError(
                    "encoder_attention_bias must be [batch, source_length] or [batch, 1, 1, source_length]"
                )
            if encoder_attention_mask is not None:
                mask = mask.masked_fill(
                    ~encoder_attention_mask.bool()[:, None, None, :],
                    torch.finfo(query.dtype).min,
                )
        elif encoder_attention_mask is not None:
            mask = encoder_attention_mask.bool()[:, None, None, :]
        if mask is not None and mask.requires_grad:
            mask = align_trainable_sdpa_bias_heads(mask, query.shape[1])
        query = ensure_sdpa_lse_for_bias_backward(query, key, value, mask)
        repeats = self.num_heads // self.num_kv_heads
        attention_kwargs = {
            "attn_mask": mask,
            "dropout_p": self.dropout if self.training else 0.0,
            "is_causal": False,
        }
        if repeats > 1 and self._sdpa_gqa_supported is not False:
            try:
                attended = F.scaled_dot_product_attention(query, key, value, enable_gqa=True, **attention_kwargs)
                self._sdpa_gqa_supported = True
            except TypeError:
                self._sdpa_gqa_supported = False
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    raise
                self._sdpa_gqa_supported = False
        if repeats == 1 or self._sdpa_gqa_supported is False:
            if repeats > 1:
                key = key.repeat_interleave(repeats, dim=1)
                value = value.repeat_interleave(repeats, dim=1)
            attended = F.scaled_dot_product_attention(query, key, value, **attention_kwargs)
        attended = attended.transpose(1, 2).reshape(batch, query_length, self.num_heads * self.head_dim)
        return self.o_proj(attended)

    @torch.no_grad()
    def prepare_memory_cache(self, encoder_hidden_states: torch.Tensor) -> None:
        self._memory_cache = tuple(value.contiguous() for value in self._project_memory(encoder_hidden_states))

    def clear_memory_cache(self) -> None:
        self._memory_cache = None


class QwenDecoderLayerWithCrossAttention(GradientCheckpointingLayer):
    """Native Qwen self-attention -> copied source attention -> native FFN."""

    def __init__(
        self,
        base_layer: nn.Module,
        config: Any,
        dropout: float,
        gate_init: float,
        initialize_from_self: bool,
        layer_index: int,
    ):
        super().__init__()
        if not 0.0 <= gate_init < 1.0:
            raise ValueError("cross gate initialization must be in [0, 1)")
        self.base_layer = base_layer
        self.cross_attn_norm = copy.deepcopy(base_layer.input_layernorm)
        self.cross_attn = QwenCopiedCrossAttention(
            base_layer.self_attn,
            base_layer.input_layernorm,
            config,
            dropout,
            initialize_from_self,
        )
        reference = base_layer.self_attn.q_proj.weight
        self.cross_attn.to(device=reference.device, dtype=reference.dtype)
        self.cross_gate = nn.Parameter(torch.tensor(math.atanh(gate_init), dtype=torch.float32))
        self.layer_index = int(layer_index)
        self.last_cross_residual_ratio: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_bias: Optional[torch.Tensor] = None,
        encoder_cross_attention_start_layer: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self.base_layer.input_layernorm(hidden_states)
        self_states, _ = self.base_layer.self_attn(
            hidden_states=normalized,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + self_states

        use_source_attention = encoder_hidden_states is not None and (
            encoder_cross_attention_start_layer is None or self.layer_index >= int(encoder_cross_attention_start_layer)
        )
        if use_source_attention:
            cross_states = self.cross_attn(
                self.cross_attn_norm(hidden_states),
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_attention_bias,
            )
            scaled = torch.tanh(self.cross_gate.float()).to(hidden_states.dtype) * cross_states
            with torch.no_grad():
                cross_rms = scaled.detach().float().square().mean().sqrt()
                hidden_rms = hidden_states.detach().float().square().mean().sqrt().clamp_min(1e-8)
                self.last_cross_residual_ratio = cross_rms / hidden_rms
            hidden_states = hidden_states + scaled

        residual = hidden_states
        hidden_states = self.base_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.base_layer.mlp(hidden_states)
        return residual + hidden_states

    @torch.no_grad()
    def prepare_cross_attention_cache(self, encoder_hidden_states: torch.Tensor) -> None:
        self.cross_attn.prepare_memory_cache(encoder_hidden_states)


class PretrainedQwenDecoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        dtype: torch.dtype,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        causal_lm, qwen_config = _load_causal_lm(model_name, dtype)
        self.model_name = str(model_name)
        self.config = qwen_config
        self.backbone = causal_lm.model
        self.lm_head = causal_lm.lm_head
        self.cross_attention_every = max(1, int(config.get("cross_attention_every", 1)))
        self.initialize_cross_from_self = bool(config.get("initialize_cross_from_self", True))
        dropout = float(config.get("cross_attention_dropout", 0.0))
        gate_init = float(config.get("cross_gate_init", 0.1))
        wrapped = []
        cross_indices = []
        base_layers = list(self.backbone.layers)
        for index, layer in enumerate(base_layers):
            self_attention = getattr(layer, "self_attn", None)
            if self_attention is None:
                raise ValueError("EviSeq requires a decoder layer with self_attn")
            if hasattr(self_attention, "layer_idx"):
                self_attention.layer_idx = index
            inject = (index + 1) % self.cross_attention_every == 0 or index == len(self.backbone.layers) - 1
            if inject:
                layer = QwenDecoderLayerWithCrossAttention(
                    layer,
                    qwen_config,
                    dropout,
                    gate_init,
                    self.initialize_cross_from_self,
                    index,
                )
                cross_indices.append(index)
            wrapped.append(layer)
        self.backbone.layers = nn.ModuleList(wrapped)
        self.cross_attention_indices = tuple(cross_indices)
        self.backbone.config.use_cache = False
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        encoder_cross_attention_start_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        if encoder_cross_attention_start_layer is not None:
            start = int(encoder_cross_attention_start_layer)
            if not 0 <= start <= len(self.backbone.layers):
                raise ValueError("encoder_cross_attention_start_layer is outside the decoder depth")
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_attention_bias=encoder_attention_bias,
            encoder_cross_attention_start_layer=encoder_cross_attention_start_layer,
        )
        return outputs.last_hidden_state, outputs.past_key_values if use_cache else None

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = bool(trainable)
        if not trainable:
            for layer in self.backbone.layers:
                if not isinstance(layer, QwenDecoderLayerWithCrossAttention):
                    continue
                for parameter in layer.cross_attn_norm.parameters():
                    parameter.requires_grad = True
                for parameter in layer.cross_attn.parameters():
                    parameter.requires_grad = True
                layer.cross_gate.requires_grad = True

    @torch.no_grad()
    def prepare_cross_attention_cache(self, encoder_hidden_states: torch.Tensor) -> None:
        self.clear_cross_attention_cache()
        for layer in self.backbone.layers:
            if isinstance(layer, QwenDecoderLayerWithCrossAttention):
                layer.prepare_cross_attention_cache(encoder_hidden_states)

    def clear_cross_attention_cache(self) -> None:
        for layer in self.backbone.layers:
            if isinstance(layer, QwenDecoderLayerWithCrossAttention):
                layer.cross_attn.clear_memory_cache()

    def cross_attention_probe_start_layer(self, probe_layers: int) -> int:
        """Return the first decoder layer used by a short source probe.

        DualBridge only needs a task/source interaction to construct its
        routing query.  Restricting that neutral probe to the final few copied
        cross-attention layers avoids repeating 4K-token K/V projections in
        every layer, while the subsequent CE/greedy pass still cross-attends
        to the fused bridge in all layers.  The integer is an explicit forward
        argument, so it is safe with gradient checkpointing.
        """

        count = int(probe_layers)
        if not 0 <= count <= len(self.cross_attention_indices):
            raise ValueError("probe_layers must be between zero and the number of copied cross-attention layers")
        if count == 0:
            return len(self.backbone.layers)
        return int(self.cross_attention_indices[-count])

    def cross_gate_mean(self) -> torch.Tensor:
        gates = [
            torch.tanh(layer.cross_gate.float())
            for layer in self.backbone.layers
            if isinstance(layer, QwenDecoderLayerWithCrossAttention)
        ]
        return torch.stack(gates).mean() if gates else next(self.parameters()).new_zeros(())

    def cross_residual_ratio_mean(self) -> torch.Tensor:
        values = [
            layer.last_cross_residual_ratio
            for layer in self.backbone.layers
            if isinstance(layer, QwenDecoderLayerWithCrossAttention) and layer.last_cross_residual_ratio is not None
        ]
        return torch.stack(values).mean() if values else next(self.parameters()).new_zeros(())
