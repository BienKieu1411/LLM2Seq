"""Legacy character-overlap copy head for AFMR.

Semantic reading is intentionally implemented in ``semantic_read.py``. This
module only prepares the lexical copy state and computes the legacy gate and
attention, which makes the ``pi_copy=g`` contract auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CopyState:
    keys: torch.Tensor
    token_ids: torch.Tensor
    mask: torch.Tensor
    bias: torch.Tensor

    def index_select(self, indices: torch.Tensor) -> "CopyState":
        return CopyState(
            self.keys.index_select(0, indices),
            self.token_ids.index_select(0, indices),
            self.mask.index_select(0, indices),
            self.bias.index_select(0, indices),
        )


@dataclass
class CopyRead:
    """Prefix-dependent copy tensors used by the probability readout."""

    query: torch.Tensor
    log_attention: torch.Tensor
    raw_gate: torch.Tensor
    g: torch.Tensor
    active: torch.Tensor


def _rms(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x.float(), (x.shape[-1],))


def _safe_log_attention(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()[:, None, :]
    floor = torch.finfo(torch.float32).min
    safe_scores = scores.float().masked_fill(~valid, floor)
    log_attention = F.log_softmax(safe_scores, dim=-1)
    active = valid.any(dim=-1, keepdim=True)
    log_attention = torch.where(active, log_attention, torch.zeros_like(log_attention))
    return log_attention.masked_fill(~valid, floor)


class GroundedCopyHead(nn.Module):
    """Copy keys plus a legacy sigmoid gate.

    The copy head never receives semantic hidden states. Its ``prepare`` method
    requires the AFMR value anchor ``H0`` explicitly so a missing value memory
    cannot silently change the copy path.
    """

    def __init__(self, hidden_size: int, key_dim: int = 128, gate_init: float = 0.05) -> None:
        super().__init__()
        if hidden_size <= 0 or key_dim <= 0 or not 0.0 < float(gate_init) < 1.0:
            raise ValueError("hidden_size/key_dim must be positive and gate_init must lie in (0,1)")
        self.hidden_size = int(hidden_size)
        self.key_dim = int(key_dim)
        self.query = nn.Linear(hidden_size, key_dim, bias=False)
        self.context_key = nn.Linear(hidden_size, key_dim, bias=False)
        self.lexical_key = nn.Linear(hidden_size, key_dim, bias=False)
        self.gate = nn.Linear(2 * key_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(float(gate_init) / (1.0 - float(gate_init))))

    def prepare(
        self,
        memory: torch.Tensor,
        source_bias: torch.Tensor,
        content_mask: torch.Tensor,
        embedding: nn.Module,
        *,
        copy_token_ids: torch.Tensor,
        copy_token_mask: torch.Tensor,
        copy_encoder_indices: torch.Tensor,
        copy_token_indices: torch.Tensor,
        copy_alignment_weights: torch.Tensor,
    ) -> CopyState:
        if memory is None:
            raise ValueError("GroundedCopyHead requires bridge.value_memory (H0)")
        if memory.ndim != 3 or source_bias.shape != memory.shape[:2] or content_mask.shape != memory.shape[:2]:
            raise ValueError("memory must be [B,S,D], source_bias/content_mask must be [B,S]")
        if copy_token_ids.ndim != 2 or copy_token_mask.shape != copy_token_ids.shape:
            raise ValueError("copy token ids/mask must be [B,W]")
        batch, width = copy_token_ids.shape
        alignment_tensors = (copy_encoder_indices, copy_token_indices, copy_alignment_weights)
        if any(t.ndim != 2 or t.shape[0] != batch for t in alignment_tensors):
            raise ValueError("copy alignment tensors must be [B,E]")
        if any(t.shape != copy_encoder_indices.shape for t in alignment_tensors):
            raise ValueError("copy alignment tensors must share edge shape [B,E]")
        if copy_encoder_indices.numel() and (
            int(copy_encoder_indices.min()) < 0 or int(copy_encoder_indices.max()) >= memory.shape[1]
        ):
            raise ValueError("copy encoder index is outside source memory")
        if copy_token_indices.numel() and (int(copy_token_indices.min()) < 0 or int(copy_token_indices.max()) >= width):
            raise ValueError("copy token index is outside candidate width")

        normalized = _rms(memory)
        projected = self.context_key(normalized.to(self.context_key.weight.dtype)).float()
        valid = content_mask.bool().gather(1, copy_encoder_indices)
        weights = copy_alignment_weights.float() * valid.float()
        totals = memory.new_zeros((batch, width), dtype=torch.float32).scatter_add(1, copy_token_indices, weights)

        def overlap_pool(values: torch.Tensor) -> torch.Tensor:
            rank = values.shape[-1]
            pooled = values.new_zeros((batch, width, rank), dtype=torch.float32).scatter_add(
                1,
                copy_token_indices[..., None].expand(-1, -1, rank),
                values.gather(1, copy_encoder_indices[..., None].expand(-1, -1, rank)).float() * weights[..., None],
            )
            return pooled / totals.clamp_min(1e-8)[..., None]

        pooled = overlap_pool(projected)
        bias = memory.new_zeros((batch, width), dtype=torch.float32).scatter_add(
            1,
            copy_token_indices,
            source_bias.float().gather(1, copy_encoder_indices) * weights,
        ) / totals.clamp_min(1e-8)

        unique_ids, inverse = torch.unique(copy_token_ids, return_inverse=True)
        lexical_bank = self.lexical_key(_rms(embedding(unique_ids)).to(self.lexical_key.weight.dtype)).float()
        lexical = lexical_bank[inverse]
        keys = _rms(pooled + lexical)
        return CopyState(keys, copy_token_ids.long(), copy_token_mask.bool() & totals.gt(0), bias)

    def attention(self, hidden: torch.Tensor, state: CopyState) -> CopyRead:
        if hidden.ndim != 3 or hidden.shape[0] != state.keys.shape[0]:
            raise ValueError("hidden must be [B,T,D] and match CopyState batch")
        query = self.query(_rms(hidden).to(self.query.weight.dtype)).float()
        scores = query @ state.keys.float().transpose(-1, -2) / math.sqrt(self.key_dim)
        scores = scores + state.bias.float()[:, None, :]
        log_attention = _safe_log_attention(scores, state.mask)
        probabilities = log_attention.exp()
        context = probabilities @ state.keys.float()
        raw_gate = self.gate(torch.cat((query, context), dim=-1).to(self.gate.weight.dtype)).float()
        active = state.mask.any(dim=-1)[:, None, None]
        g = torch.sigmoid(raw_gate).where(active, torch.zeros_like(raw_gate))
        return CopyRead(query, log_attention, raw_gate, g, active)

    def distribution(self, hidden: torch.Tensor, state: CopyState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy-compatible log attention, log copy and log generation."""

        read = self.attention(hidden, state)
        floor = torch.finfo(torch.float32).min
        log_g = torch.where(read.active, F.logsigmoid(read.raw_gate), torch.full_like(read.raw_gate, floor))
        log_generate = torch.where(read.active, F.logsigmoid(-read.raw_gate), torch.zeros_like(read.raw_gate))
        return read.log_attention, log_g, log_generate

    @staticmethod
    def copy_log_prob(read: CopyRead, state: CopyState, vocab_size: int) -> torch.Tensor:
        """Marginalize duplicate candidate IDs into a vocabulary distribution."""

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        result = read.log_attention.new_zeros((*read.log_attention.shape[:2], vocab_size))
        result.scatter_add_(
            -1,
            state.token_ids[:, None, :].expand(-1, read.log_attention.shape[1], -1),
            read.log_attention.exp(),
        )
        floor = torch.finfo(torch.float32).min
        return result.clamp_min(torch.finfo(torch.float32).tiny).log().masked_fill(result.eq(0), floor)

    def mix_logits(self, hidden: torch.Tensor, logits: torch.Tensor, state: CopyState) -> torch.Tensor:
        """Legacy copy-only output used by the AFMR ``base_only`` endpoint."""

        read = self.attention(hidden, state)
        log_copy = torch.where(read.active, F.logsigmoid(read.raw_gate), torch.full_like(read.raw_gate, -float("inf")))
        log_generate = torch.where(read.active, F.logsigmoid(-read.raw_gate), torch.zeros_like(read.raw_gate))
        log_pcopy = self.copy_log_prob(read, state, logits.shape[-1])
        z = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        mixed = torch.logaddexp(logits.float() - z + log_generate, log_pcopy + log_copy) + z
        return torch.where(read.active, mixed, logits.float())


__all__ = ["CopyRead", "CopyState", "GroundedCopyHead"]
