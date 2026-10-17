"""Decoder context-bound checks shared by rollout and policy scoring."""

from __future__ import annotations

import math
from typing import Any

import torch


def validate_context(decoder: Any, prompt_width: int, max_new_tokens: int | float | torch.Tensor) -> None:
    """Reject requests beyond a decoder's declared positional capacity.

    Some decoder configurations omit a finite ``max_position_embeddings``;
    those models retain their own context behavior and are left unchecked here.
    """

    limit_value = getattr(getattr(decoder, "config", None), "max_position_embeddings", None)
    try:
        limit = float(limit_value)
    except (TypeError, ValueError, OverflowError):
        return
    if not math.isfinite(limit) or limit <= 0:
        return

    if isinstance(max_new_tokens, torch.Tensor):
        if max_new_tokens.numel() == 0:
            raise ValueError("max_new_tokens must contain at least one value")
        budget = float(max_new_tokens.detach().float().max().item())
    else:
        budget = float(max_new_tokens)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("max_new_tokens must be finite and nonnegative")

    required = int(prompt_width) + math.ceil(budget)
    if required > limit:
        raise ValueError(
            f"Requested decoder context ({prompt_width} prompt + {math.ceil(budget)} generated tokens = "
            f"{required}) exceeds decoder.config.max_position_embeddings={int(limit)}"
        )
