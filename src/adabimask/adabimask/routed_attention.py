"""Attention wrapper that routes each source layer through a mask policy."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .mask_policy import LayerMaskPolicy


def make_bidirectional_mask(
    attention_mask: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Remove only the causal constraint while retaining key padding.

    Qwen3 with the SDPA backend passes either a 4-D additive causal mask or
    ``None`` to every layer. The last causal query can see all non-padding
    keys, so repeating its row yields the equivalent padding-only mask without
    relying on a private Transformers mask utility.
    """

    batch_size, query_length = hidden_states.shape[:2]
    dtype = hidden_states.dtype
    device = hidden_states.device

    if attention_mask is None:
        return torch.zeros(
            (batch_size, 1, query_length, query_length),
            dtype=dtype,
            device=device,
        )

    if attention_mask.ndim == 2:
        key_length = attention_mask.shape[-1]
        if not dtype.is_floating_point:
            raise TypeError(f"hidden_states must have floating dtype, got {dtype}")
        full_mask = torch.zeros((batch_size, 1, query_length, key_length), dtype=dtype, device=device)
        valid_keys = attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
        return full_mask.masked_fill(~valid_keys, torch.finfo(dtype).min)

    if attention_mask.ndim != 4:
        raise ValueError(f"Expected 2-D or 4-D attention mask, got shape {tuple(attention_mask.shape)}")

    final_query_row = attention_mask[..., -1:, :]
    return final_query_row.expand(-1, -1, query_length, -1)


def _with_attention_mask(kwargs: Dict[str, Any], mask: torch.Tensor) -> Dict[str, Any]:
    routed = dict(kwargs)
    if "attention_mask" not in routed:
        raise TypeError("RoutedSelfAttention requires the backbone layer to pass attention_mask as a keyword argument.")
    routed["attention_mask"] = mask
    return routed


def _mix_attention_results(
    causal_result: Any,
    bidirectional_result: Any,
    gate: torch.Tensor,
) -> Any:
    if not isinstance(causal_result, tuple) or not isinstance(bidirectional_result, tuple):
        return torch.lerp(causal_result, bidirectional_result, gate.to(causal_result.dtype))

    mixed = []
    for causal_value, bidirectional_value in zip(causal_result, bidirectional_result):
        if isinstance(causal_value, torch.Tensor) and isinstance(bidirectional_value, torch.Tensor):
            mixed.append(torch.lerp(causal_value, bidirectional_value, gate.to(causal_value.dtype)))
        else:
            # Qwen/SDPA returns None for attention weights. Preserve that value.
            mixed.append(causal_value)
    return tuple(mixed)


class RoutedSelfAttention(nn.Module):
    """Wrap one pretrained self-attention block without copying its weights."""

    def __init__(self, base_attention: nn.Module, layer_index: int, policy: LayerMaskPolicy):
        super().__init__()
        self.base_attention = base_attention
        self.layer_index = int(layer_index)
        # Keep one registered policy under the encoder instead of duplicating it
        # under all attention wrappers in the state dict.
        object.__setattr__(self, "_policy", policy)

    @property
    def policy(self) -> LayerMaskPolicy:
        return object.__getattribute__(self, "_policy")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("Could not find hidden_states tensor in attention call")

        route = self.policy.route(self.layer_index)
        if route == "causal":
            return self.base_attention(*args, **kwargs)

        past_key_value = kwargs.get("past_key_value")
        if past_key_value is not None:
            raise RuntimeError(
                "Bidirectional source attention does not support KV-cache; call encoder with use_cache=False"
            )

        causal_mask = kwargs.get("attention_mask")
        bidirectional_mask = make_bidirectional_mask(causal_mask, hidden_states)
        bidirectional_kwargs = _with_attention_mask(kwargs, bidirectional_mask)
        if route == "bidirectional":
            return self.base_attention(*args, **bidirectional_kwargs)

        causal_result = self.base_attention(*args, **kwargs)
        bidirectional_result = self.base_attention(*args, **bidirectional_kwargs)
        gate = self.policy.gate_for_layer(self.layer_index)
        return _mix_attention_results(causal_result, bidirectional_result, gate)
