"""Stable tensor contracts shared by AFMR model components."""

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
    source_bias: torch.Tensor
    controller: torch.Tensor
    value_memory: Optional[torch.Tensor] = None
    copy_state: Optional[CopyState] = None

    def index_select(self, indices: torch.Tensor) -> BridgeState:
        return BridgeState(
            memory=self.memory.index_select(0, indices),
            memory_mask=self.memory_mask.index_select(0, indices),
            content_mask=self.content_mask.index_select(0, indices),
            source_bias=self.source_bias.index_select(0, indices),
            controller=self.controller.index_select(0, indices),
            value_memory=None if self.value_memory is None else self.value_memory.index_select(0, indices),
            copy_state=None if self.copy_state is None else self.copy_state.index_select(indices),
        )


@dataclass
class AFMROutput:
    logits: Optional[torch.Tensor]
    loss_ce: Optional[torch.Tensor]
    loss: Optional[torch.Tensor]
    bridge: BridgeState
