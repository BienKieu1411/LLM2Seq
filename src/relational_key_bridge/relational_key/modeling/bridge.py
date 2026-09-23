"""Directional adjacent-source relations used only as cross-attention keys."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import validate_architecture
from .outputs import BridgeState, EncoderState


class RelationalKeyBridge(nn.Module):
    """Keep source values/copy fixed while making retrieval sensitive to local pairs.

    The encoder hidden states are already contextual. The extra pair operator is
    a finite-budget directional inductive bias, not a source of new information.
    """

    def __init__(self, encoder_hidden: int, decoder_hidden: int, config: dict):
        super().__init__()
        validate_architecture(config)
        self.bridge_mode = str(config.get("bridge_mode", "relational_key"))
        self.direct_projection: nn.Module = nn.Identity()
        if encoder_hidden != decoder_hidden:
            self.direct_projection = nn.Linear(encoder_hidden, decoder_hidden, bias=False)
            nn.init.orthogonal_(self.direct_projection.weight)
        if self.bridge_mode == "direct_projection":
            return

        rank = int(config.get("relation_rank", 256))
        self.key_gate_max = float(config.get("key_gate_max", 0.20))
        initial = float(config.get("key_gate_init", 0.10))
        self.residual_reference_rms = float(config.get("residual_reference_rms", 1.0))
        # Isolate extra initialization from subsequent dataloader/dropout draws.
        with torch.random.fork_rng(devices=[]):
            self.source_norm = nn.RMSNorm(encoder_hidden, eps=1e-6)
            self.center_down = nn.Linear(encoder_hidden, rank, bias=False)
            self.left_down = nn.Linear(encoder_hidden, rank, bias=False)
            self.right_down = nn.Linear(encoder_hidden, rank, bias=False)
            self.relation_up = nn.Linear(2 * rank, decoder_hidden, bias=False)
            nn.init.orthogonal_(self.relation_up.weight, gain=float(config.get("output_init_gain", 1.0)))
            self.key_gate_raw = nn.Parameter(torch.tensor(math.log(initial / (self.key_gate_max - initial))))

    @property
    def requires_alignment(self) -> bool:
        return False

    def _relation(self, final: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        # Prefix and padding are zeroed *before* shifting, so their features
        # never enter the first/last content token's relation.
        source = F.rms_norm(
            final.float(), (final.shape[-1],), self.source_norm.weight.float(), self.source_norm.eps
        ).to(self.center_down.weight.dtype)
        source = source.masked_fill(~content[..., None], 0.0)
        center = self.center_down(source)
        left = self.left_down(source)
        right = self.right_down(source)
        left_neighbor = F.pad(left[:, :-1], (0, 0, 1, 0))
        right_neighbor = F.pad(right[:, 1:], (0, 0, 0, 1))
        left_valid = F.pad(content[:, 1:] & content[:, :-1], (1, 0))
        right_valid = F.pad(content[:, :-1] & content[:, 1:], (0, 1))
        left_pair = F.silu(center * left_neighbor) * left_valid[..., None]
        right_pair = F.silu(center * right_neighbor) * right_valid[..., None]
        return self.relation_up(torch.cat((left_pair, right_pair), dim=-1))

    def _capped_residual(self, residual: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        anchor_rms = anchor.detach().float().square().mean(-1, keepdim=True).sqrt()
        denominator = (self.residual_reference_rms**2 + residual.float().square().mean(-1, keepdim=True)).sqrt()
        return anchor_rms * residual.float() / denominator

    def forward(
        self,
        encoder_state: EncoderState,
        decoder_embedding: nn.Module | None = None,
        **alignment: torch.Tensor,
    ) -> BridgeState:
        del decoder_embedding, alignment
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
        anchor = self.direct_projection(final.to(projection_dtype)).masked_fill(~memory_mask[..., None], 0.0)
        if self.bridge_mode == "direct_projection":
            return BridgeState(anchor, memory_mask, content, anchor, anchor)

        residual = self._relation(final, content)
        residual = self._capped_residual(residual, anchor).masked_fill(~content[..., None], 0.0)
        gate = self.key_gate_max * self.key_gate_raw.float().sigmoid()
        key_memory = (anchor.float() + gate * residual).to(anchor.dtype)
        return BridgeState(key_memory, memory_mask, content, anchor, anchor)
