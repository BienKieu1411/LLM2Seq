"""Ordered decoder-token evidence for cross-attention retrieval and values."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import validate_architecture
from .outputs import BridgeState, EncoderState


class CrossTokenizerOrderedValueBridge(nn.Module):
    def __init__(self, encoder_hidden: int, decoder_hidden: int, config: dict):
        super().__init__()
        validate_architecture(config)
        self.bridge_mode = str(config.get("bridge_mode", "cross_tokenizer_ordered_value"))
        # This is the same module, name, and initialization order in both modes.
        # Equal widths retain the baseline's identity projection.
        self.direct_projection: nn.Module = nn.Identity()
        if encoder_hidden != decoder_hidden:
            self.direct_projection = nn.Linear(encoder_hidden, decoder_hidden, bias=False)
            nn.init.orthogonal_(self.direct_projection.weight)
        if self.bridge_mode == "direct_projection":
            return
        self.value_gate_max = float(config.get("value_gate_max", 0.20))
        self.key_gate_max = float(config.get("key_gate_max", 0.20))
        self.residual_reference_rms = float(config.get("residual_reference_rms", 1.0))
        rank = int(config.get("lexical_rank", 256))
        kernel = int(config.get("phrase_kernel", 3))
        initial = float(config.get("value_gate_init", 0.10))
        key_initial = float(config.get("key_gate_init", 0.12))
        self.value_gate_mode = str(config.get("value_gate_mode", "global"))
        # Adding XOV must not advance the RNG used for later data/dropout draws.
        with torch.random.fork_rng(devices=[]):
            self.lexical_norm = nn.RMSNorm(decoder_hidden, eps=1e-6)
            self.lexical_down = nn.Linear(decoder_hidden, rank, bias=False)
            # Conv1d's independent left/right coefficients retain direction.
            self.phrase_conv = (
                nn.Conv1d(rank, rank, kernel, padding=kernel // 2, groups=rank, bias=False)
                if kernel == 3
                else nn.Identity()
            )
            self.lexical_up = nn.Linear(rank, decoder_hidden, bias=False)
            nn.init.orthogonal_(self.lexical_up.weight, gain=float(config.get("output_init_gain", 1.0)))
            self.value_gate_raw = nn.Parameter(torch.tensor(math.log(initial / (self.value_gate_max - initial))))
            self.key_gate_raw = nn.Parameter(torch.tensor(math.log(key_initial / (self.key_gate_max - key_initial))))
            if self.value_gate_mode == "source_lexical":
                self.source_gate = nn.Linear(decoder_hidden, 1, bias=False)
                self.lexical_gate = nn.Linear(rank, 1, bias=False)
                nn.init.zeros_(self.source_gate.weight)
                nn.init.zeros_(self.lexical_gate.weight)

    @property
    def requires_alignment(self) -> bool:
        return self.bridge_mode == "cross_tokenizer_ordered_value"

    def _ordered_features(self, compact: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if isinstance(self.phrase_conv, nn.Identity):
            return compact
        weights = self.phrase_conv.weight[:, 0, :]
        adjacent = valid[:, 1:] & valid[:, :-1] & (positions[:, 1:] == positions[:, :-1] + 1)
        left = F.pad(compact[:, :-1] * adjacent[..., None], (0, 0, 1, 0))
        right = F.pad(compact[:, 1:] * adjacent[..., None], (0, 0, 0, 1))
        return left * weights[:, 0] + compact * weights[:, 1] + right * weights[:, 2]

    @staticmethod
    def _reverse_scatter(
        ordered: torch.Tensor,
        content: torch.Tensor,
        token_mask: torch.Tensor,
        encoder_indices: torch.Tensor,
        token_indices: torch.Tensor,
        alignment_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reverse decoder-normalized overlaps, renormalizing at each encoder token."""
        batch, source_length = content.shape
        rank = ordered.shape[-1]
        valid = content.gather(1, encoder_indices) & token_mask.gather(1, token_indices)
        weights = alignment_weights.float().masked_fill(~valid, 0.0)
        selected = ordered.gather(1, token_indices[..., None].expand(-1, -1, rank)).float()
        pooled = selected.new_zeros(batch, source_length, rank).scatter_add(
            1, encoder_indices[..., None].expand(-1, -1, rank), selected * weights[..., None]
        )
        totals = weights.new_zeros(batch, source_length).scatter_add(1, encoder_indices, weights)
        aligned = totals.gt(0) & content
        pooled = (pooled / totals.clamp_min(1e-8)[..., None]).masked_fill(~aligned[..., None], 0.0)
        return pooled, aligned

    def _unit_capped(self, residual: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # RMS(r / sqrt(reference^2 + mean(r^2))) is <= 1, including at r=0.
        # The detached scale creates no second gradient route into the anchor.
        reference = memory.detach().float().square().mean(-1, keepdim=True).sqrt()
        denominator = (self.residual_reference_rms**2 + residual.float().square().mean(-1, keepdim=True)).sqrt()
        return reference * (residual.float() / denominator)

    def forward(
        self,
        encoder_state: EncoderState,
        decoder_embedding: nn.Module,
        **alignment: torch.Tensor,
    ) -> BridgeState:
        final = encoder_state.final
        if final.ndim != 3:
            raise ValueError("encoder_state.final must be [batch, source_tokens, hidden]")
        if encoder_state.attention_mask.shape != final.shape[:2] or encoder_state.content_mask.shape != final.shape[:2]:
            raise ValueError("encoder masks must match encoder_state.final")
        memory_mask = encoder_state.attention_mask.bool()
        content = encoder_state.content_mask.bool() & memory_mask
        projection_dtype = (
            self.direct_projection.weight.dtype if isinstance(self.direct_projection, nn.Linear) else torch.float32
        )
        memory = self.direct_projection(final.to(projection_dtype)).masked_fill(~memory_mask[..., None], 0.0)
        if not self.requires_alignment:
            return BridgeState(memory, memory_mask, content, memory, memory)
        required = {
            "copy_token_ids",
            "copy_token_mask",
            "copy_token_positions",
            "copy_encoder_indices",
            "copy_token_indices",
            "copy_alignment_weights",
        }
        if missing := required - alignment.keys():
            raise ValueError(f"XOV source alignment is missing: {sorted(missing)}")
        token_ids = alignment["copy_token_ids"]
        token_mask = alignment["copy_token_mask"].bool()
        edges_valid = content.gather(1, alignment["copy_encoder_indices"]) & alignment["copy_alignment_weights"].gt(0)
        covered = torch.zeros_like(token_ids).scatter_add(1, alignment["copy_token_indices"], edges_valid.long()).gt(0)
        token_mask = token_mask & covered
        # Embed vocabulary types once and gather only rank-wide representations;
        # no [batch, decoder_source_length, decoder_hidden] lexical tensor exists.
        unique_ids, inverse = torch.unique(token_ids, return_inverse=True)
        normalized = self.lexical_norm(decoder_embedding(unique_ids).to(self.lexical_norm.weight.dtype))
        bank = self.lexical_down(normalized.to(self.lexical_down.weight.dtype))
        compact = bank[inverse].masked_fill(~token_mask[..., None], 0.0)
        ordered = self._ordered_features(compact, alignment["copy_token_positions"], token_mask)
        ordered = ordered.masked_fill(~token_mask[..., None], 0.0)
        pooled, aligned = self._reverse_scatter(
            F.silu(ordered),
            content,
            token_mask,
            alignment["copy_encoder_indices"],
            alignment["copy_token_indices"],
            alignment["copy_alignment_weights"],
        )
        residual = self.lexical_up(pooled.to(self.lexical_up.weight.dtype)).float()
        residual = self._unit_capped(residual, memory).masked_fill(~aligned[..., None], 0.0)
        gate_logit = self.value_gate_raw.float()
        if self.value_gate_mode == "source_lexical":
            source_features = F.rms_norm(memory.float(), (memory.shape[-1],))
            lexical_features = F.rms_norm(pooled.float(), (pooled.shape[-1],))
            gate_logit = (
                gate_logit
                + self.source_gate(source_features.to(self.source_gate.weight.dtype)).float()
                + self.lexical_gate(lexical_features.to(self.lexical_gate.weight.dtype)).float()
            )
        gate = self.value_gate_max * gate_logit.sigmoid()
        key_gate = self.key_gate_max * self.key_gate_raw.float().sigmoid()
        keys = (memory.float() + key_gate * residual).to(memory.dtype)
        values = (memory.float() + gate * residual).to(memory.dtype)
        # Copy sees the original projected encoder states, not the lexical key route.
        return BridgeState(keys, memory_mask, content, values, memory)
