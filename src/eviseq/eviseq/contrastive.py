"""Target-free source--prompt contrastive learning for EviSeq.

The decoder query is collected at the final fixed prompt token, before the
shifted decoder input exposes any reference-summary token.  The loss therefore
cannot be solved by teacher-forced target text: it rewards a decoder that has
actually read the matching encoder memory through cross-attention.

Training applies the symmetric loss over the complete optimizer window with
the exact two-pass GradCache decomposition in ``training.py``. Gradient
accumulation therefore enlarges the negative set instead of merely summing
independent physical-batch losses.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean_pool(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool valid source tokens in FP32.

    Long-document runs pool up to 4,096 BF16 states.  Accumulating before the
    cast loses avoidable precision exactly at the representation consumed by
    InfoNCE, so only the inexpensive pooled path is promoted to FP32.
    """

    expanded = mask.unsqueeze(-1).float()
    states = hidden_states.float()
    return (states * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1.0)


def masked_last_pool(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select the final valid source token."""

    if hidden_states.ndim != 3 or mask.ndim != 2 or hidden_states.shape[:2] != mask.shape:
        raise ValueError("masked_last_pool expects [B, S, D] states and a matching [B, S] mask")
    if not bool(mask.bool().any(dim=1).all()):
        raise ValueError("Every source example must contain at least one valid token")
    positions = mask.long().sum(dim=1) - 1
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, positions]


def last_prompt_states(decoder_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gather the state predicting the first supervised target token.

    At this position the shifted decoder input still contains the final fixed
    prompt token.  No reference-summary token is visible to the query.
    """

    if decoder_states.ndim != 3 or labels.ndim != 2 or decoder_states.shape[:2] != labels.shape:
        raise ValueError("decoder_states and labels must be matching [B, T, ...] tensors")
    supervised = labels.ne(-100)
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("Every example must contain at least one supervised target token")
    positions = supervised.long().argmax(dim=1)
    rows = torch.arange(decoder_states.shape[0], device=decoder_states.device)
    return decoder_states[rows, positions]


def exact_duplicate_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mark off-diagonal examples with byte-identical tokenized sources.

    Duplicate sources are alternative positives, not valid InfoNCE negatives.
    The comparison is intentionally exact; near-duplicate mining would add a
    second model and an unaudited threshold to the training objective.
    """

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must be matching [B, S] tensors")
    valid_ids = input_ids.masked_fill(~attention_mask.bool(), 0)
    same_ids = valid_ids[:, None, :].eq(valid_ids[None, :, :]).all(dim=-1)
    same_mask = attention_mask[:, None, :].eq(attention_mask[None, :, :]).all(dim=-1)
    duplicates = same_ids & same_mask
    duplicates.fill_diagonal_(False)
    return duplicates


class SourcePromptAlignmentHead(nn.Module):
    """Project source memory and target-free decoder prompts to one space."""

    def __init__(self, hidden_size: int, projection_size: int = 256, pooling: str = "mean_last"):
        super().__init__()
        self.pooling = str(pooling)
        if self.pooling not in {"mean", "mean_last"}:
            raise ValueError("contrastive pooling must be 'mean' or 'mean_last'")
        self.pool_gate: Optional[nn.Parameter]
        if self.pooling == "mean_last":
            self.pool_gate = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        else:
            self.register_parameter("pool_gate", None)
        self.source_projection = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, projection_size, bias=False),
        )
        self.prompt_projection = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, projection_size, bias=False),
        )
        nn.init.xavier_uniform_(self.source_projection[-1].weight)
        nn.init.xavier_uniform_(self.prompt_projection[-1].weight)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        prompt_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if memory.ndim != 3:
            raise ValueError("EviSeq contrastive learning expects one [B, S, D] source memory")
        source = masked_mean_pool(memory, memory_mask)
        if self.pool_gate is not None:
            last = masked_last_pool(memory, memory_mask)
            last_weight = torch.sigmoid(self.pool_gate.float()).to(source.dtype)
            source = (1.0 - last_weight) * source + last_weight * last
        if prompt_states.ndim != 2:
            raise ValueError("prompt_states must be [B, D]")
        return {
            "source_repr": F.normalize(self.source_projection(source).float(), dim=-1),
            "prompt_repr": F.normalize(self.prompt_projection(prompt_states).float(), dim=-1),
        }


def info_nce_loss(
    source_repr: torch.Tensor,
    prompt_repr: torch.Tensor,
    temperature: float,
    duplicate_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric in-batch InfoNCE and source-retrieval accuracy."""

    if source_repr.shape != prompt_repr.shape or source_repr.ndim != 2:
        raise ValueError("source_repr and prompt_repr must be matching [B, D] tensors")
    if temperature <= 0.0:
        raise ValueError("contrastive temperature must be positive")
    batch_size = source_repr.shape[0]
    if batch_size <= 1:
        zero = source_repr.sum() * 0.0
        return zero, zero.detach()
    logits = prompt_repr @ source_repr.T / float(temperature)
    if duplicate_mask is not None:
        if duplicate_mask.shape != (batch_size, batch_size):
            raise ValueError("duplicate_mask must be [B, B]")
        logits = logits.masked_fill(duplicate_mask, torch.finfo(logits.dtype).min)
    labels = torch.arange(batch_size, device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    accuracy = logits.detach().argmax(dim=1).eq(labels).float().mean()
    return loss, accuracy
