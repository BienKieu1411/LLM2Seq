"""Contiguous source regions used by the shared cross-attention/copy router."""

from __future__ import annotations

import torch


def pool_source_regions(
    memory: torch.Tensor, content_mask: torch.Tensor, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return region vectors and a per-source-position region ID (-1 outside content)."""
    if width <= 0:
        raise ValueError("evidence region width must be positive")
    if memory.ndim != 3 or content_mask.shape != memory.shape[:2]:
        raise ValueError("memory and content mask must have matching source dimensions")
    batch, source_length, hidden = memory.shape
    valid = content_mask.bool()
    ordinal = valid.long().cumsum(dim=1) - 1
    region_ids = torch.where(valid, ordinal.div(width, rounding_mode="floor"), -1)
    region_count = max(1, (source_length + width - 1) // width)
    destination = region_ids.clamp_min(0)
    counts = memory.new_zeros(batch, region_count, dtype=torch.float32).scatter_add(1, destination, valid.float())
    regions = memory.new_zeros(batch, region_count, hidden, dtype=torch.float32).scatter_add(
        1,
        destination[..., None].expand(-1, -1, hidden),
        memory.float() * valid[..., None],
    )
    regions = regions / counts.clamp_min(1.0)[..., None]
    return regions.to(memory.dtype), region_ids
