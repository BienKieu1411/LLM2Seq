"""Separate retrieval, value, and copy memory contracts for RelationalKey."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .grounded_copy import CopyState


@dataclass
class EncoderState:
    final: torch.Tensor
    taps: tuple[torch.Tensor, ...]
    attention_mask: torch.Tensor
    content_mask: torch.Tensor


@dataclass
class BridgeState:
    memory: torch.Tensor
    memory_mask: torch.Tensor
    content_mask: torch.Tensor
    value_memory: torch.Tensor
    copy_memory: torch.Tensor
    copy_state: Optional[CopyState] = None

    def to_dtype(self, dtype: torch.dtype) -> BridgeState:
        memory = self.memory.to(dtype)
        value_memory = memory if self.value_memory is self.memory else self.value_memory.to(dtype)
        copy_memory = memory if self.copy_memory is self.memory else self.copy_memory.to(dtype)
        return BridgeState(
            memory,
            self.memory_mask,
            self.content_mask,
            value_memory,
            copy_memory,
            self.copy_state,
        )

    def index_select(self, indices: torch.Tensor) -> BridgeState:
        """Reorder/duplicate every source tensor together for cache compaction."""
        memory = self.memory.index_select(0, indices)
        return BridgeState(
            memory,
            self.memory_mask.index_select(0, indices),
            self.content_mask.index_select(0, indices),
            memory if self.value_memory is self.memory else self.value_memory.index_select(0, indices),
            memory if self.copy_memory is self.memory else self.copy_memory.index_select(0, indices),
            None if self.copy_state is None else self.copy_state.index_select(indices),
        )


@dataclass
class RelationalKeyOutput:
    logits: Optional[torch.Tensor]
    loss_ce: Optional[torch.Tensor]
    loss: Optional[torch.Tensor]
    bridge: BridgeState
