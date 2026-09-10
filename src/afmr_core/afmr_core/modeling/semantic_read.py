"""Flat source-conditioned semantic reader used by AFMR.

The reader is deliberately small: it exposes the legacy graph exactly, keeps source
states cacheable, and applies one numerically stable relative RMS bound to the
residual.  Probability routing lives in :mod:`dual_readout`, so this module
cannot silently modify the copy distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms(x: torch.Tensor) -> torch.Tensor:
    """RMS over the final dimension, always reduced in FP32."""

    x32 = x.float()
    return torch.sqrt(x32.square().mean(dim=-1, keepdim=True))


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMS-normalize a vector while retaining its final dimension."""

    return F.rms_norm(x.float(), (x.shape[-1],))


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log probabilities and probabilities without all-masked NaNs."""

    if scores.ndim != 3 or mask.ndim != 2 or scores.shape[0] != mask.shape[0] or scores.shape[-1] != mask.shape[-1]:
        raise ValueError("semantic scores must be [B,T,S] and mask must be [B,S]")
    valid = mask.bool()[:, None, :]
    floor = torch.finfo(torch.float32).min
    safe_scores = scores.float().masked_fill(~valid, floor)
    log_probs = F.log_softmax(safe_scores, dim=-1)
    has_source = valid.any(dim=-1, keepdim=True)
    log_probs = torch.where(has_source, log_probs, torch.zeros_like(log_probs))
    log_probs = log_probs.masked_fill(~valid, floor)
    probs = log_probs.exp().masked_fill(~valid, 0.0)
    return log_probs, probs


def smooth_relative_rms_cap(
    delta: torch.Tensor,
    hidden: torch.Tensor,
    rho: float = 0.10,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Smoothly bound residual RMS by ``rho * RMS(hidden)``.

    ``tanh(x) / x`` has limit one at zero, so the operation preserves a tiny
    residual's direction while saturating large projections.  The calculation
    is FP32 and never creates an all-zero/NaN branch for a zero residual.
    """

    if rho <= 0 or not math.isfinite(float(rho)):
        raise ValueError("rho must be finite and positive")
    if eps <= 0 or not math.isfinite(float(eps)):
        raise ValueError("eps must be finite and positive")
    hidden32, delta32 = hidden.float(), delta.float()
    target = (float(rho) * rms(hidden32)).clamp_min(float(eps))
    residual_rms = rms(delta32)
    ratio = residual_rms / target
    safe_ratio = ratio.clamp_min(float(eps))
    scale = torch.tanh(safe_ratio) / safe_ratio
    scale = torch.where(residual_rms.eq(0), torch.ones_like(scale), scale)
    return delta32 * scale


@dataclass
class SemanticState:
    """Source-side tensors cached once per document/source batch."""

    key_memory: torch.Tensor
    value_memory: torch.Tensor
    source_mask: torch.Tensor
    source_bias: torch.Tensor
    key_source: str
    value_source: str
    prior_scale: float

    def index_select(self, indices: torch.Tensor) -> "SemanticState":
        return SemanticState(
            self.key_memory.index_select(0, indices),
            self.value_memory.index_select(0, indices),
            self.source_mask.index_select(0, indices),
            self.source_bias.index_select(0, indices),
            self.key_source,
            self.value_source,
            self.prior_scale,
        )


class SemanticReader(nn.Module):
    """Independent flat source read with the exact legacy RMS ordering."""

    def __init__(
        self,
        hidden_size: int,
        rank: int = 128,
        *,
        key_source: str = "H0",
        value_source: str = "H0",
        semantic_prior_scale: float = 1.0,
        max_relative_rms: float = 0.10,
        cap_mode: str = "smooth_relative_rms",
        inner_gate: bool = False,
        gate_init: float = 0.05,
        output_init: str = "tiny_rms_1e-3",
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or rank <= 0:
            raise ValueError("hidden_size and rank must be positive")
        if key_source not in {"H0", "M"} or value_source != "H0":
            raise ValueError("AFMR semantic reader supports key_source H0/M and value_source H0")
        if not 0.0 < float(max_relative_rms) < math.inf:
            raise ValueError("max_relative_rms must be finite and positive")
        if cap_mode not in {"smooth_relative_rms", "legacy"}:
            raise ValueError("cap_mode must be smooth_relative_rms or legacy")
        if inner_gate and not 0.0 < float(gate_init) < 1.0:
            raise ValueError("gate_init must lie in (0,1)")
        if output_init not in {"zero", "tiny_rms_1e-3"}:
            raise ValueError("output_init must be zero or tiny_rms_1e-3")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.key_source = key_source
        self.value_source = value_source
        self.semantic_prior_scale = float(semantic_prior_scale)
        self.max_relative_rms = float(max_relative_rms)
        self.cap_mode = str(cap_mode)
        self.inner_gate = bool(inner_gate)
        self.output_init = output_init
        self._tiny_calibrated = False
        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.key = nn.Linear(hidden_size, rank, bias=False)
        self.value = nn.Linear(hidden_size, rank, bias=False)
        self.output = nn.Linear(rank, hidden_size, bias=False)
        self.gate = nn.Linear(2 * rank, 1, bias=True) if self.inner_gate else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, math.log(float(gate_init) / (1.0 - float(gate_init))))
        if output_init == "zero":
            nn.init.zeros_(self.output.weight)
        else:
            # The calibration method can tighten this to an exact RMS ratio;
            # this scale keeps the first candidate update close to the endpoint.
            nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-3 / math.sqrt(rank))

    @staticmethod
    def _source_norm(memory: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(memory.float(), (memory.shape[-1],))

    @staticmethod
    def _project(layer: nn.Linear, values: torch.Tensor) -> torch.Tensor:
        """Run a projection after matching the layer dtype, then expose FP32."""

        return layer(values.to(dtype=layer.weight.dtype)).float()

    def _cap_residual(self, delta: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if self.cap_mode == "smooth_relative_rms":
            return smooth_relative_rms_cap(delta, hidden, self.max_relative_rms)
        cap = (
            self.max_relative_rms
            * torch.linalg.vector_norm(hidden.float(), dim=-1, keepdim=True)
            / math.sqrt(hidden.shape[-1])
        )
        denominator = torch.sqrt(cap.square() + delta.float().square().mean(dim=-1, keepdim=True) + 1.0e-12)
        return delta.float() * cap / denominator

    def prepare(
        self,
        *,
        H0: torch.Tensor,
        M: torch.Tensor,
        source_mask: torch.Tensor,
        source_bias: torch.Tensor,
    ) -> SemanticState:
        """Project/cache source memory; ``H0`` is mandatory for values."""

        if H0 is None:
            raise ValueError("SemanticReader requires bridge.value_memory (H0)")
        if M is None:
            raise ValueError("SemanticReader requires AFMR retrieval memory (M)")
        if H0.ndim != 3 or M.ndim != 3 or H0.shape != M.shape:
            raise ValueError("H0 and M must both be [B,S,D] with matching shape")
        if source_mask.shape != H0.shape[:2] or source_bias.shape != H0.shape[:2]:
            raise ValueError("source_mask and source_bias must be [B,S]")
        key_memory = H0 if self.key_source == "H0" else M
        key_norm = self._source_norm(key_memory)
        value_norm = self._source_norm(H0)
        keys = _rms_norm(self._project(self.key, key_norm))
        values = self._project(self.value, value_norm)
        return SemanticState(
            keys,
            values,
            source_mask.bool(),
            source_bias.float(),
            self.key_source,
            self.value_source,
            self.semantic_prior_scale,
        )

    def read(
        self,
        hidden: torch.Tensor,
        state: SemanticState,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Read semantic context and return ``hs`` plus diagnostics."""

        if hidden.ndim != 3:
            raise ValueError("hidden must be [B,T,D]")
        if hidden.shape[0] != state.key_memory.shape[0] or hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden batch/width does not match semantic state")
        query = self._project(self.query, _rms_norm(hidden))
        scores = torch.matmul(query, state.key_memory.float().transpose(-1, -2)) / math.sqrt(self.rank)
        scores = scores + float(state.prior_scale) * state.source_bias.float()[:, None, :]
        log_attention, attention = _masked_softmax(scores, state.source_mask)
        context = torch.matmul(attention, state.value_memory.float())
        normalized_context = _rms_norm(context)
        if self.gate is None:
            gate = torch.ones_like(normalized_context[..., :1])
        else:
            gate = torch.sigmoid(self._project(self.gate, torch.cat((query, normalized_context), dim=-1)))
        delta_raw = gate * self._project(self.output, normalized_context)
        delta = self._cap_residual(delta_raw, hidden)
        evidence = state.source_mask.any(dim=-1)[:, None, None].float()
        delta = delta * evidence
        hs = (hidden.float() + delta).to(hidden.dtype)
        diagnostics = {
            "q": query,
            "u": normalized_context,
            "attention": attention,
            "log_attention": log_attention,
            "gate": gate,
            "delta_raw": delta_raw,
            "delta": delta,
            "evidence": evidence.expand(-1, hidden.shape[1], -1),
        }
        return hs, diagnostics

    @torch.no_grad()
    def calibrate_tiny(self, hidden: torch.Tensor, context: torch.Tensor, target_ratio: float = 1.0e-3) -> float:
        """Rescale ``output`` on an unlabeled calibration batch.

        Returns the measured RMS ratio after rescaling.  This method is
        intentionally independent of labels and is deterministic given the
        caller's seed and batch.
        """

        if target_ratio <= 0 or not math.isfinite(float(target_ratio)):
            raise ValueError("target_ratio must be finite and positive")
        raw = self._project(self.output, context.float())
        numerator = float(rms(raw).mean())
        denominator = float(rms(hidden).mean())
        if numerator <= 0.0 or denominator <= 0.0:
            return 0.0
        scale = float(target_ratio) * denominator / numerator
        self.output.weight.mul_(scale)
        self._tiny_calibrated = True
        measured = float(rms(self._project(self.output, context.float())).mean() / rms(hidden).mean().clamp_min(1e-12))
        return measured


__all__ = ["SemanticReader", "SemanticState", "rms", "smooth_relative_rms_cap"]
