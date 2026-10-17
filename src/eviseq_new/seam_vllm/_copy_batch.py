"""Reuse immutable copy inputs while the ordered active request batch is stable."""

from __future__ import annotations

import torch
from eviseq_afmr.modeling.grounded_copy import CopyState


def stack_copy_states(states: list[CopyState | None]) -> CopyState | None:
    if not states or all(state is None for state in states):
        return None
    if any(state is None for state in states):
        raise ValueError("SEAM batch mixes requests with and without required grounded-copy state")
    typed = [state for state in states if state is not None]
    if len(typed) == 1:
        return typed[0]
    max_width = max(state.token_ids.shape[-1] for state in typed)
    first = typed[0]
    batch = len(typed)
    keys = first.keys.new_zeros(batch, max_width, first.keys.shape[-1])
    token_ids = first.token_ids.new_zeros(batch, max_width)
    mask = first.mask.new_zeros(batch, max_width)
    bias = first.bias.new_zeros(batch, max_width)
    widths = [state.token_ids.shape[-1] for state in typed]
    # Use four tensor-list calls rather than dispatching four copy_ operations
    # per request. Heterogeneous widths may still take PyTorch's per-tensor
    # CUDA path, so this reduces Python dispatch without claiming fused copies.
    for name, destination in (("keys", keys), ("token_ids", token_ids), ("mask", mask), ("bias", bias)):
        torch._foreach_copy_(
            [destination[row, :width] for row, width in enumerate(widths)],
            [getattr(state, name)[0] for state in typed],
        )
    return CopyState(keys, token_ids, mask, bias)


class CopyBatchCache:
    """One-entry cache; source CopyStates must remain immutable after preparation.

    Strong references prevent identity reuse. Any change to member identity or
    ordering rebuilds the batch. Removing a request explicitly clears the cache.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.members: tuple[CopyState | None, ...] = ()
        self.packed: CopyState | None = None

    def get(self, states: list[CopyState | None]) -> CopyState | None:
        if len(states) == len(self.members) and all(current is old for current, old in zip(states, self.members)):
            return self.packed
        packed = stack_copy_states(states)
        self.members, self.packed = tuple(states), packed
        return packed
