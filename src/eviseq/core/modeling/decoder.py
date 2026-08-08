"""Full pretrained Qwen3 decoder with conventional copied cross-attention."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:  # pragma: no cover - compatibility guard for older Transformers
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
    in the original LLM2Seq decoder. Unlike the old scratch decoder, Q/K/V/O
    begin in a pretrained Qwen coordinate system.
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
        # PyTorch 2.1+ can run grouped-query attention without materialising
        # num_heads / num_kv_heads copies of every 4K-token memory at every
        # generated token.  Keep a compatibility fallback for older runtimes.
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
            if key.shape[0] != batch:
                raise RuntimeError("Stale cross-attention cache; rebuild it for the current batch")
        mask = None
        if encoder_attention_bias is not None:
            mask = encoder_attention_bias.to(query.dtype)[:, None, None, :]
            if encoder_attention_mask is not None:
                mask = mask.masked_fill(
                    ~encoder_attention_mask.bool()[:, None, None, :],
                    torch.finfo(query.dtype).min,
                )
        elif encoder_attention_mask is not None:
            mask = encoder_attention_mask.bool()[:, None, None, :]
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
            except (RuntimeError, TypeError):
                # Some older PyTorch builds expose no GQA kernel for a given
                # device/backend.  The expanded implementation is equivalent.
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
        memory_bank_count: int,
        layer_index: int,
        total_layers: int,
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
        self.memory_bank_count = int(memory_bank_count)
        if self.memory_bank_count > 1:
            if self.memory_bank_count != 3:
                raise ValueError("Depth routing currently requires three memory banks")
            depth = float(layer_index) / max(1, total_layers - 1)
            lexical = max(0.0, 1.0 - 2.0 * depth)
            semantic = max(0.0, 1.0 - abs(2.0 * depth - 1.0))
            summary = max(0.0, 2.0 * depth - 1.0)
            prior = torch.tensor([lexical, semantic, summary], dtype=torch.float32)
            prior = prior.clamp_min(0.05)
            prior = prior / prior.sum()
            self.memory_router_logits = nn.Parameter(prior.log())
        else:
            self.register_parameter("memory_router_logits", None)
        self.last_cross_residual_ratio: Optional[torch.Tensor] = None

    def routing_weights(self) -> torch.Tensor:
        if self.memory_router_logits is None:
            return self.cross_gate.new_ones(1)
        return F.softmax(self.memory_router_logits.float(), dim=0)

    def route_memory(self, memory: torch.Tensor) -> torch.Tensor:
        if memory.ndim == 3:
            if self.memory_bank_count != 1:
                raise ValueError("Depth-routed decoder expected [B, 3, S, D] memory")
            return memory
        if memory.ndim != 4 or memory.shape[1] != self.memory_bank_count:
            raise ValueError(
                f"Expected [B, {self.memory_bank_count}, S, D] source memory, received {tuple(memory.shape)}"
            )
        weights = self.routing_weights().to(memory.dtype)
        return torch.einsum("k,bksd->bsd", weights, memory)

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

        if encoder_hidden_states is not None:
            source_memory = encoder_hidden_states
            if source_memory.ndim == 4 and (self.training or self.cross_attn._memory_cache is None):
                source_memory = self.route_memory(source_memory)
            cross_states = self.cross_attn(
                self.cross_attn_norm(hidden_states),
                source_memory,
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
        self.cross_attn.prepare_memory_cache(self.route_memory(encoder_hidden_states))


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
        self.memory_bank_count = int(config.get("memory_bank_count", 1))
        dropout = float(config.get("cross_attention_dropout", 0.0))
        gate_init = float(config.get("cross_gate_init", 0.1))
        wrapped = []
        cross_indices = []
        base_layers = list(self.backbone.layers)
        for index, layer in enumerate(base_layers):
            self_attention = getattr(layer, "self_attn", None)
            if self_attention is None:
                raise ValueError("LLM2Seq-v2 requires a Qwen decoder layer with self_attn")
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
                    self.memory_bank_count,
                    index,
                    len(base_layers),
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
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_attention_bias=encoder_attention_bias,
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
                if layer.memory_router_logits is not None:
                    layer.memory_router_logits.requires_grad = True

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

    def memory_routing_mean(self) -> torch.Tensor:
        values = [
            layer.routing_weights()
            for layer in self.backbone.layers
            if isinstance(layer, QwenDecoderLayerWithCrossAttention)
        ]
        if not values:
            return next(self.parameters()).new_ones(1)
        return torch.stack(values).mean(dim=0)
