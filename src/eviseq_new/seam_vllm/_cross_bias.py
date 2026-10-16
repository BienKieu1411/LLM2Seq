"""Represent the additive source prior in padded cross-attention coordinates."""

from __future__ import annotations

import math

import torch


def pad_cross_query(query: torch.Tensor, padded_dim: int) -> torch.Tensor:
    head_dim = query.shape[-1]
    if padded_dim < head_dim + 2:
        raise ValueError("SEAM cross-attention needs two source-bias coordinates")
    padded = query.new_zeros(*query.shape[:-1], padded_dim)
    padded[..., :head_dim] = query
    padded[..., head_dim : head_dim + 2] = 1.0
    return padded


def pad_cross_key(key: torch.Tensor, source_bias: torch.Tensor, padded_dim: int) -> torch.Tensor:
    head_dim = key.shape[-1]
    if padded_dim < head_dim + 2 or source_bias.shape != key.shape[:1]:
        raise ValueError("SEAM cross key/bias dimensions are inconsistent")
    padded = key.new_zeros(*key.shape[:-1], padded_dim)
    padded[..., :head_dim] = key
    # Native AFMR casts the additive bias to the query dtype. Split its scaled
    # representation into two components so BF16 rounding does not dominate it.
    scaled = source_bias.to(key.dtype).float() * math.sqrt(head_dim)
    high = scaled.to(key.dtype)
    low = (scaled - high.float()).to(key.dtype)
    padded[..., head_dim] = high[:, None]
    padded[..., head_dim + 1] = low[:, None]
    return padded
