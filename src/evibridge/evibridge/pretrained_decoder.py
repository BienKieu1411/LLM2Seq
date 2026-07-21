"""Pretrained shallow Qwen decoder with gated encoder cross-attention."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_layers import GradientCheckpointingLayer

from .backbone import load_text_causal_lm, torch_dtype


def normalized_layer_indices(num_source_layers: int, num_decoder_layers: int) -> Tuple[int, ...]:
    """Select depth-uniform pretrained blocks without architecture surgery."""

    if not 1 <= num_decoder_layers <= num_source_layers:
        raise ValueError(
            f"num_decoder_layers must be in [1, {num_source_layers}], got {num_decoder_layers}"
        )
    if num_decoder_layers == 1:
        return (num_source_layers - 1,)
    indices = tuple(
        round(index * (num_source_layers - 1) / (num_decoder_layers - 1))
        for index in range(num_decoder_layers)
    )
    if len(set(indices)) != len(indices):  # pragma: no cover - defensive for unusual rounding rules
        raise RuntimeError(f"Layer selection produced duplicates: {indices}")
    return indices


class HeadRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float() * torch.rsqrt(
            hidden_states.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(hidden_states.dtype)


class QwenCrossAttention(nn.Module):
    """GQA cross-attention initialized from a native Qwen self-attention."""

    def __init__(self, config: Any, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.num_heads))
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.q_norm = HeadRMSNorm(self.head_dim, eps)
        self.k_norm = HeadRMSNorm(self.head_dim, eps)
        self.dropout = float(dropout)
        self._cached_key: Optional[torch.Tensor] = None
        self._cached_value: Optional[torch.Tensor] = None

    @torch.no_grad()
    def initialize_from_self_attention(self, source: nn.Module) -> None:
        for name in ("k_proj", "v_proj", "o_proj"):
            source_weight = getattr(source, name).weight
            target_weight = getattr(self, name).weight
            if source_weight.shape != target_weight.shape:
                raise ValueError(
                    f"Cannot initialize cross-attention {name}: {source_weight.shape} != {target_weight.shape}"
                )
            target_weight.copy_(source_weight)

        source_q = source.q_proj.weight
        if source_q.shape[0] < self.q_proj.weight.shape[0]:
            raise ValueError(f"Q projection is too small: {source_q.shape} -> {self.q_proj.weight.shape}")
        # Qwen3.5 packs an output gate after Q in q_proj; Qwen3 does not.
        self.q_proj.weight.copy_(source_q[: self.q_proj.weight.shape[0]])

        for name in ("q_norm", "k_norm"):
            source_norm = getattr(source, name, None)
            if source_norm is None or not hasattr(source_norm, "weight"):
                continue
            weight = source_norm.weight.detach().float()
            # Qwen3.5 stores a zero-centered residual scale (1 + weight), while
            # Qwen3 stores the scale directly.
            converted = 1.0 + weight if float(weight.abs().mean()) < 0.5 else weight
            getattr(self, name).weight.copy_(converted)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        key_length = encoder_hidden_states.shape[1]
        query = self.q_proj(hidden_states).view(batch_size, query_length, self.num_heads, self.head_dim)
        if self._cached_key is not None and self._cached_value is not None and not self.training:
            key = self._cached_key
            value = self._cached_value
            if key.shape[0] != batch_size or key.shape[2] != key_length:
                raise RuntimeError("Stale cross-attention memory cache; prepare it for the current source")
        else:
            key = self.k_proj(encoder_hidden_states).view(
                batch_size, key_length, self.num_kv_heads, self.head_dim
            )
            value = self.v_proj(encoder_hidden_states).view(
                batch_size, key_length, self.num_kv_heads, self.head_dim
            )
            key = self.k_norm(key).transpose(1, 2)
            value = value.transpose(1, 2)
        query = self.q_norm(query).transpose(1, 2)
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)

        attention_mask = None
        if encoder_attention_mask is not None:
            attention_mask = encoder_attention_mask.to(torch.bool)[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, query_length, -1)
        return self.o_proj(attended)

    @torch.no_grad()
    def prepare_memory_cache(self, encoder_hidden_states: torch.Tensor) -> None:
        batch_size, key_length, _ = encoder_hidden_states.shape
        key = self.k_proj(encoder_hidden_states).view(
            batch_size, key_length, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(encoder_hidden_states).view(
            batch_size, key_length, self.num_kv_heads, self.head_dim
        )
        self._cached_key = self.k_norm(key).transpose(1, 2).contiguous()
        self._cached_value = value.transpose(1, 2).contiguous()

    def clear_memory_cache(self) -> None:
        self._cached_key = None
        self._cached_value = None


class CrossAttentionInjectedLayer(GradientCheckpointingLayer):
    """Native token mixer -> gated cross-attention -> native FFN."""

    def __init__(self, base_layer: nn.Module, config: Any, source_attention: nn.Module, dropout: float):
        super().__init__()
        self.base_layer = base_layer
        self.cross_attn_norm = copy.deepcopy(base_layer.post_attention_layernorm)
        self.cross_attn = QwenCrossAttention(config, dropout=dropout)
        reference_weight = source_attention.q_proj.weight
        self.cross_attn.to(device=reference_weight.device, dtype=reference_weight.dtype)
        self.cross_attn.initialize_from_self_attention(source_attention)
        self.cross_gate = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        mixed_states = self.base_layer.input_layernorm(hidden_states)
        if getattr(self.base_layer, "block_type", None) == "linear_attention":
            mixed_states = self.base_layer.linear_attn(
                hidden_states=mixed_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            mixed_states, _ = self.base_layer.self_attn(
                hidden_states=mixed_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        hidden_states = residual + mixed_states

        if encoder_hidden_states is not None:
            cross_states = self.cross_attn(
                self.cross_attn_norm(hidden_states),
                encoder_hidden_states,
                encoder_attention_mask,
            )
            hidden_states = hidden_states + torch.tanh(self.cross_gate).to(hidden_states.dtype) * cross_states

        residual = hidden_states
        hidden_states = self.base_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.base_layer.mlp(hidden_states)
        return residual + hidden_states


def _nearest_attention_layer(layers: Sequence[nn.Module], layer_index: int) -> nn.Module:
    candidates: List[Tuple[int, nn.Module]] = []
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            candidates.append((index, attention))
    if not candidates:
        raise RuntimeError("The pretrained decoder has no full self-attention layer for cross initialization")
    return min(candidates, key=lambda item: (abs(item[0] - layer_index), item[0]))[1]


class PretrainedQwenDecoder(nn.Module):
    """Depth-compressed native Qwen decoder; only cross-attention is new."""

    def __init__(self, model_name: str, decoder_config: Dict[str, Any], dtype_name: str):
        super().__init__()
        causal_lm, config = load_text_causal_lm(
            model_name,
            dtype=torch_dtype(dtype_name),
            attn_implementation="sdpa",
        )
        self.config = config
        self.backbone = causal_lm.model
        self.lm_head = causal_lm.lm_head
        all_layers = list(self.backbone.layers)
        source_layer_count = len(all_layers)
        requested_layers = int(decoder_config.get("num_layers", 16))
        indices = normalized_layer_indices(source_layer_count, requested_layers)
        dropout = float(decoder_config.get("cross_attention_dropout", 0.0))

        selected_types = []
        wrapped_layers = []
        original_types = list(getattr(config, "layer_types", ["full_attention"] * source_layer_count))
        for new_index, source_index in enumerate(indices):
            layer = all_layers[source_index]
            source_attention = _nearest_attention_layer(all_layers, source_index)
            token_mixer = getattr(layer, "self_attn", getattr(layer, "linear_attn", None))
            if token_mixer is not None and hasattr(token_mixer, "layer_idx"):
                token_mixer.layer_idx = new_index
            if token_mixer is not None and hasattr(token_mixer, "layer_type"):
                token_mixer.layer_type = original_types[source_index]
            wrapped_layers.append(
                CrossAttentionInjectedLayer(layer, config, source_attention, dropout=dropout)
            )
            selected_types.append(original_types[source_index])

        self.layer_indices = indices
        self.backbone.layers = nn.ModuleList(wrapped_layers)
        self.config.num_hidden_layers = requested_layers
        self.backbone.config.num_hidden_layers = requested_layers
        if hasattr(self.config, "layer_types"):
            self.config.layer_types = selected_types
            self.backbone.config.layer_types = selected_types
        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False
        if bool(decoder_config.get("gradient_checkpointing", True)):
            self.backbone.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        **_: Any,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return outputs.last_hidden_state, outputs.past_key_values

    def set_backbone_trainable(self, enabled: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = bool(enabled)
        # New cross-attention remains trainable during the interface warm-up.
        for layer in self.backbone.layers:
            for parameter in layer.cross_attn_norm.parameters():
                parameter.requires_grad = True
            for parameter in layer.cross_attn.parameters():
                parameter.requires_grad = True
            layer.cross_gate.requires_grad = True

    @torch.no_grad()
    def prepare_cross_attention_cache(self, encoder_hidden_states: torch.Tensor) -> None:
        """Project static encoder K/V once instead of once per generated token."""

        self.clear_cross_attention_cache()
        for layer in self.backbone.layers:
            layer.cross_attn.prepare_memory_cache(encoder_hidden_states)

    def clear_cross_attention_cache(self) -> None:
        for layer in self.backbone.layers:
            layer.cross_attn.clear_memory_cache()

    @property
    def embed_tokens(self) -> nn.Module:
        return self.backbone.embed_tokens

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)
