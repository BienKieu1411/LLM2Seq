"""Small shape helpers shared with dependency-light tests."""

from __future__ import annotations


def padded_attention_head_dim(head_dim: int, alignment: int = 8) -> int:
    """Reserve two compensated bias coordinates and align for attention kernels."""

    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((head_dim + 2 + alignment - 1) // alignment) * alignment
