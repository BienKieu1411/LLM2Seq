"""Bound HF encoder/FP32 bridge activations independently of decode concurrency."""

from __future__ import annotations

import torch

from ._tensors import _as_2d


def encoder_batches(encoder_input_ids, fields, device, max_tokens: int):
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("encoder_max_batch_tokens must be a positive integer")
    ids = _as_2d(encoder_input_ids, "encoder_input_ids", device, torch.long)
    batch, width = ids.shape
    if not batch or not width:
        raise ValueError("SEAM encoder batch must contain non-empty source rows")
    # Count padded positions: they also consume encoder/bridge activation memory.
    # A source wider than this budget still runs alone, without truncation.
    rows_per_batch = max(1, max_tokens // width)
    dtypes = {
        "encoder_attention_mask": torch.bool,
        "source_content_mask": torch.bool,
        "copy_token_ids": torch.long,
        "copy_token_mask": torch.bool,
        "copy_encoder_indices": torch.long,
        "copy_token_indices": torch.long,
        "copy_alignment_weights": torch.float32,
        "requested_output_tokens": torch.float32,
        "bridge_prompt_ids": torch.long,
        "bridge_prompt_mask": torch.bool,
    }
    normalized = {}
    for name, value in fields.items():
        if value is not None and name in dtypes:
            value = _as_2d(value, name, device, dtypes[name])
            if value.shape[0] != batch:
                raise ValueError(f"SEAM field {name} must have one row per source")
        normalized[name] = value
    for start in range(0, batch, rows_per_batch):
        end = start + rows_per_batch
        yield {
            "encoder_input_ids": ids[start:end],
            **{
                name: value[start:end] if value is not None and name in dtypes else value
                for name, value in normalized.items()
            },
        }
