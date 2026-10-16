"""Batch variable-length encoder and decoder fields without losing padding masks."""

from __future__ import annotations

from typing import Any

import torch


def pack_bridge_memory(memory, value_memory, source_bias):
    """Only duplicate memory when the trained bridge has a separate value anchor."""
    bias = source_bias.unsqueeze(-1).to(memory.dtype)
    return torch.cat((memory, bias) if value_memory is None else (memory, value_memory, bias), dim=-1)


def unpack_bridge_memory(packed, hidden: int):
    # The appended bias makes the packed row width odd. A slice can have
    # stride(-1)==1 while its BF16 row starts are unaligned for vLLM's vector
    # loads in RMSNorm. Materialize once per encoder prefill, before any layer.
    if packed.shape[-1] == hidden + 1:
        memory = packed[:, :hidden].contiguous()
        return memory, memory, packed[:, hidden].float().contiguous()
    if packed.shape[-1] == 2 * hidden + 1:
        return (
            packed[:, :hidden].contiguous(),
            packed[:, hidden : 2 * hidden].contiguous(),
            packed[:, 2 * hidden].float().contiguous(),
        )
    raise ValueError(f"SEAM encoder output has width {packed.shape[-1]}, expected {hidden + 1} or {2 * hidden + 1}")


def _as_2d(value: Any, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, (list, tuple)) and value and not isinstance(value[0], (int, float, bool)):
        rows = [torch.as_tensor(row, device=device, dtype=dtype).reshape(-1) for row in value]
        width = max((row.numel() for row in rows), default=0)
        fill = False if dtype == torch.bool else 0
        output = torch.full((len(rows), width), fill, device=device, dtype=dtype)
        for index, row in enumerate(rows):
            output[index, : row.numel()] = row
        tensor = output
    else:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        while tensor.ndim > 2 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"SEAM field {name} must be rank 2 after batching; got {tuple(tensor.shape)}")
    return tensor


def bridge_prompt_batch(default_ids, prompt_ids, prompt_mask, batch: int, device):
    if prompt_ids is None:
        if prompt_mask is not None:
            raise ValueError("SEAM bridge prompt mask requires prompt IDs")
        ids = default_ids.to(device=device).unsqueeze(0).expand(batch, -1)
        return ids, torch.ones_like(ids, dtype=torch.bool)
    if prompt_mask is None:
        raise ValueError("SEAM per-request bridge prompt IDs require a padding mask")
    ids = _as_2d(prompt_ids, "bridge_prompt_ids", device, torch.long)
    mask = _as_2d(prompt_mask, "bridge_prompt_mask", device, torch.bool)
    if ids.shape != mask.shape or ids.shape[0] != batch:
        raise ValueError("SEAM bridge prompt IDs/masks must match the source batch")
    if not mask.any(dim=-1).all():
        raise ValueError("SEAM bridge prompt must contain at least one valid token per source")
    return ids, mask
