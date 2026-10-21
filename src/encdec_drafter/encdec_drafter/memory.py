from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn


def ordered_masked_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    budget: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Expected hidden_states [batch, source, width] and attention_mask [batch, source]")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("Source hidden states and mask have incompatible shapes")
    if budget < 1:
        raise ValueError("Pooling budget must be positive")

    batch_size, _, width = hidden_states.shape
    pooled = hidden_states.new_zeros((batch_size, budget, width))
    pooled_mask = torch.zeros((batch_size, budget), dtype=torch.bool, device=hidden_states.device)
    positions = torch.zeros((batch_size, budget), dtype=torch.long, device=hidden_states.device)
    for batch_index in range(batch_size):
        valid = hidden_states[batch_index][attention_mask[batch_index].bool()]
        length = valid.shape[0]
        if length == 0:
            raise ValueError("Source memory cannot be empty or fully masked")
        bins = min(length, budget)
        for slot in range(bins):
            start = (slot * length) // bins
            stop = ((slot + 1) * length) // bins
            pooled[batch_index, slot] = valid[start:stop].mean(dim=0)
            pooled_mask[batch_index, slot] = True
            positions[batch_index, slot] = slot
    return pooled, pooled_mask, positions


@dataclass
class LayerSourceKV:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class PreparedDraftSource:
    layer_kv: List[LayerSourceKV]
    mask: torch.Tensor
    fingerprint: str
    source_tokens: int
    retained_tokens: int


def sinusoidal_positions(positions: torch.Tensor, width: int, dtype: torch.dtype) -> torch.Tensor:
    half = (width + 1) // 2
    scale = torch.exp(
        torch.arange(half, device=positions.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1))
    )
    phases = positions.float().unsqueeze(-1) * scale
    embedded = torch.cat([phases.sin(), phases.cos()], dim=-1)[..., :width]
    return embedded.to(dtype=dtype)


def bytes_for_tensor(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def cache_bytes(cache) -> int:
    total = 0
    for cache_part in (cache.self_attention_cache, cache.cross_attention_cache):
        for layer in cache_part.layers:
            total += bytes_for_tensor(getattr(layer, "keys", None))
            total += bytes_for_tensor(getattr(layer, "values", None))
    return total
