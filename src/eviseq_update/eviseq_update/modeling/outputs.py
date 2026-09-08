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


@dataclass
class LossStatistics:
    """Unnormalised differentiable losses used by the trainer/DDP reducer."""

    ce_sum: torch.Tensor
    gold_token_count: torch.Tensor
    copy_sum: torch.Tensor
    semantic_sum: torch.Tensor
    evidence_count: torch.Tensor


@dataclass
class DecoderResult:
    logits: Optional[torch.Tensor]
    past_key_values: Optional[object]
    loss_ce: Optional[torch.Tensor]
    loss_statistics: Optional[LossStatistics]


@dataclass
class AFMROutput:
    logits: Optional[torch.Tensor]
    loss_ce: Optional[torch.Tensor]
    loss: Optional[torch.Tensor]
    bridge: BridgeState
    loss_statistics: Optional[LossStatistics] = None
