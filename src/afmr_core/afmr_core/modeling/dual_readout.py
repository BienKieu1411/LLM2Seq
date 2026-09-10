"""Probability-space readout and stable mixture likelihood for AFMR."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .grounded_copy import CopyRead, CopyState, GroundedCopyHead


def _log_with_zero(value: torch.Tensor) -> torch.Tensor:
    floor = torch.finfo(torch.float32).min
    value32 = value.float()
    safe = value32.clamp_min(torch.finfo(torch.float32).tiny).log()
    return safe.masked_fill(value32 <= 0, floor)


def normalized_copy_entropy(log_attention: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Entropy divided by log(number of valid candidates), shape ``[B,T,1]``."""

    valid = mask.bool()[:, None, :]
    probabilities = log_attention.exp().masked_fill(~valid, 0.0)
    entropy = -(probabilities * log_attention.masked_fill(~valid, 0.0)).sum(-1, keepdim=True)
    count = valid.sum(-1, keepdim=True).clamp_min(2).float().log()
    return (entropy / count).clamp(0.0, 1.0)


@dataclass
class ReadState:
    h: torch.Tensor
    hs: torch.Tensor
    z0: torch.Tensor
    zs: torch.Tensor
    copy_log_attention: torch.Tensor
    copy_log_prob: torch.Tensor
    raw_copy_gate: torch.Tensor
    g: torch.Tensor
    semantic_evidence: torch.Tensor
    alpha: torch.Tensor
    pi_base: torch.Tensor
    pi_sem: torch.Tensor
    pi_copy: torch.Tensor
    log_p0: torch.Tensor
    log_ps: torch.Tensor
    log_p: torch.Tensor
    gauge: torch.Tensor
    output_logits: torch.Tensor
    diagnostics: dict[str, torch.Tensor]

    @property
    def probability(self) -> torch.Tensor:
        return self.log_p.exp()


def target_log_probability(state: ReadState, labels: torch.Tensor) -> torch.Tensor:
    """Gather the final mixture log probability for each target position."""

    if labels.ndim != 2 or labels.shape[:2] != state.log_p.shape[:2]:
        raise ValueError("labels must be [B,T] matching ReadState")
    safe_labels = labels.clamp_min(0).long()
    return state.log_p.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)


def copy_target_log_probability(copy_read: CopyRead, copy_state: CopyState, labels: torch.Tensor) -> torch.Tensor:
    """Gather copy mass for target IDs without allocating a vocabulary tensor."""

    safe_labels = labels.clamp_min(0).long()
    matches = copy_state.token_ids[:, None, :] == safe_labels[..., None]
    mass = (copy_read.log_attention.exp() * matches.float()).sum(-1)
    floor = torch.finfo(torch.float32).min
    safe = mass.float().clamp_min(torch.finfo(torch.float32).tiny).log()
    return safe.masked_fill(mass <= 0, floor)


def _mixture_log_probability(
    pi_base: torch.Tensor,
    pi_sem: torch.Tensor,
    pi_copy: torch.Tensor,
    target_log_p0: torch.Tensor,
    target_log_ps: torch.Tensor,
    target_log_copy: torch.Tensor,
) -> torch.Tensor:
    """Apply the one probability-space mixture kernel to gathered targets."""

    routes = (pi_base, pi_sem, pi_copy)
    targets = (target_log_p0, target_log_ps, target_log_copy)
    terms = [
        _log_with_zero(route.squeeze(-1) if route.shape[-1] == 1 else route) + target
        for route, target in zip(routes, targets)
    ]
    return torch.logsumexp(torch.stack(terms, dim=-1), dim=-1)


def _mixture_nll_from_targets(
    pi_base: torch.Tensor,
    pi_sem: torch.Tensor,
    pi_copy: torch.Tensor,
    target_log_p0: torch.Tensor,
    target_log_ps: torch.Tensor,
    target_log_copy: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return NLL from gathered branch targets for dense and streamed paths."""

    valid = labels.ne(-100)
    log_mix = _mixture_log_probability(pi_base, pi_sem, pi_copy, target_log_p0, target_log_ps, target_log_copy)
    return -log_mix.masked_select(valid).sum(), valid.sum()


def mixture_nll_from_state(state: ReadState, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unnormalised NLL and valid-token count from one ``ReadState``."""

    if labels.shape != state.log_p.shape[:2]:
        raise ValueError("labels must be [B,T] matching ReadState")
    safe_labels = labels.clamp_min(0).long()
    target_log_p0 = state.log_p0.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    target_log_ps = state.log_ps.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    target_log_copy = state.copy_log_prob.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return _mixture_nll_from_targets(
        state.pi_base,
        state.pi_sem,
        state.pi_copy,
        target_log_p0,
        target_log_ps,
        target_log_copy,
        labels,
    )


class DualReadout(nn.Module):
    """Two shared-vocabulary readouts plus a copy-mass-aware router."""

    MODES = {
        "copy_mass_preserving_capped_simplex",
        "independent_capped_simplex",
        "legacy_copy_mixture",
        "hidden_interpolation",
        "constant_alpha",
    }

    def __init__(
        self,
        hidden_size: int,
        rank: int = 128,
        *,
        mode: str = "copy_mass_preserving_capped_simplex",
        alpha_max: float = 0.20,
        generate_reserve: float = 0.05,
        alpha_init: float = 0.05,
        base_floor: float = 0.05,
        hidden_lambda: float = 0.10,
        detach_copy_route_features: bool = True,
        hard_source_fallback: bool = True,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown readout mode: {mode}")
        if not 0.0 < alpha_max <= 1.0 or not 0.0 <= generate_reserve < 1.0:
            raise ValueError("alpha_max must be in (0,1] and reserve in [0,1)")
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must lie in (0, alpha_max)")
        if not 0.0 <= base_floor < 1.0:
            raise ValueError("base_floor must be in [0,1)")
        if hidden_lambda < 0.0:
            raise ValueError("hidden_lambda must be non-negative")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.mode = mode
        self.alpha_max = float(alpha_max)
        self.generate_reserve = float(generate_reserve)
        self.base_floor = float(base_floor)
        self.hidden_lambda = float(hidden_lambda)
        self.detach_copy_route_features = bool(detach_copy_route_features)
        self.hard_source_fallback = bool(hard_source_fallback)
        # q and u are rank-dimensional; raw gate and entropy are scalar.
        feature_dim = 2 * int(rank) + 2
        self.router = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.router.weight)
        nn.init.constant_(self.router.bias, math.log(float(alpha_init) / (float(alpha_max) - float(alpha_init))))
        self.independent_router = nn.Linear(feature_dim, 3)
        nn.init.zeros_(self.independent_router.weight)
        nn.init.zeros_(self.independent_router.bias)

    def _routes(
        self,
        features: torch.Tensor,
        evidence: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g_route = g.detach()
        if self.mode == "legacy_copy_mixture":
            # Reproduce legacy's two-way mixture: once semantic evidence exists,
            # its vocabulary branch receives all non-copy mass.  With no
            # source evidence, preserve the base LM branch and the hard
            # source-empty fallback used by the legacy copy head.
            evidence_float = evidence.float().clamp(0.0, 1.0)
            pi_copy = g
            pi_sem = (1.0 - g) * evidence_float
            pi_base = (1.0 - g) * (1.0 - evidence_float)
            return pi_base, pi_sem, pi_copy, torch.zeros_like(pi_sem)
        if self.mode == "independent_capped_simplex":
            floor = torch.finfo(torch.float32).min
            residual = self.independent_router(features.float())
            base_prior = _log_with_zero((1.0 - g_route).clamp(0.0, 1.0))
            copy_prior = _log_with_zero(g_route.clamp(0.0, 1.0))
            semantic_prior = torch.full_like(base_prior, -20.0)
            valid = torch.cat((1.0 - g_route, evidence, g_route), dim=-1)
            logits = torch.cat((base_prior, semantic_prior, copy_prior), dim=-1) + residual
            logits = logits.masked_fill(valid.eq(0), floor)
            q = torch.softmax(logits.float(), dim=-1)
            pi_base = self.base_floor + (1.0 - self.base_floor) * q[..., 0:1]
            pi_sem = (1.0 - self.base_floor) * q[..., 1:2]
            pi_copy = (1.0 - self.base_floor) * q[..., 2:3]
            return pi_base, pi_sem, pi_copy, torch.zeros_like(pi_sem)

        alpha_raw = self.alpha_max * torch.sigmoid(self.router(features.float())) * evidence
        cap = (1.0 - g_route - self.generate_reserve).clamp_min(0.0)
        alpha = torch.minimum(alpha_raw, cap)
        if self.mode == "constant_alpha":
            alpha = torch.minimum(torch.full_like(alpha_raw, self.alpha_max * 0.25) * evidence, cap)
        pi_copy = g
        pi_sem = alpha
        pi_base = (1.0 - g - alpha).clamp_min(0.0)
        return pi_base, pi_sem, pi_copy, alpha_raw - alpha

    def _routing(
        self,
        h: torch.Tensor,
        semantic_diagnostics: dict[str, torch.Tensor],
        copy_read: CopyRead,
        copy_state: CopyState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = semantic_diagnostics.get("q", h.new_zeros((*h.shape[:2], self.rank))).float()
        u = semantic_diagnostics.get("u", h.new_zeros((*h.shape[:2], self.rank))).float()
        entropy = normalized_copy_entropy(copy_read.log_attention, copy_state.mask)
        raw_gate = copy_read.raw_gate.detach() if self.detach_copy_route_features else copy_read.raw_gate
        entropy = entropy.detach() if self.detach_copy_route_features else entropy
        features = torch.cat((q, u, raw_gate, entropy), dim=-1)
        evidence = semantic_diagnostics.get("evidence", h.new_zeros((*h.shape[:2], 1))).float()
        if evidence.shape != h.shape[:2] + (1,):
            raise ValueError("semantic evidence must be [B,T,1]")
        return (*self._routes(features, evidence, copy_read.g), evidence)

    def forward(
        self,
        h: torch.Tensor,
        hs: torch.Tensor,
        semantic_diagnostics: dict[str, torch.Tensor],
        copy_read: CopyRead,
        copy_state: CopyState,
        lm_head: nn.Module,
        *,
        return_logits: bool = True,
    ) -> ReadState:
        if h.shape != hs.shape or h.ndim != 3:
            raise ValueError("h and hs must be matching [B,T,D] tensors")
        z0 = lm_head(h).float()
        zs = lm_head(hs).float()
        log_p0 = z0 - torch.logsumexp(z0, dim=-1, keepdim=True)
        log_ps = zs - torch.logsumexp(zs, dim=-1, keepdim=True)
        log_pcopy = GroundedCopyHead.copy_log_prob(copy_read, copy_state, z0.shape[-1])
        pi_base, pi_sem, pi_copy, cap_gap, evidence = self._routing(h, semantic_diagnostics, copy_read, copy_state)
        log_components = torch.stack(
            (
                _log_with_zero(pi_base) + log_p0,
                _log_with_zero(pi_sem) + log_ps,
                _log_with_zero(pi_copy) + log_pcopy,
            ),
            dim=-2,
        )
        log_p = torch.logsumexp(log_components, dim=-2)
        gauge = torch.logsumexp(z0, dim=-1, keepdim=True)
        output_logits = log_p + gauge
        if self.hard_source_fallback:
            inactive = ~(copy_read.active | evidence.bool())
            output_logits = torch.where(inactive, z0, output_logits)
        entropy = normalized_copy_entropy(copy_read.log_attention, copy_state.mask).detach()
        diagnostics = {
            "copy_entropy_norm": entropy,
            "cap_gap": cap_gap,
            "log_pcopy": log_pcopy,
            "log_p0": log_p0,
            "log_ps": log_ps,
            "prob_sum": log_p.exp().sum(-1, keepdim=True),
        }
        if not return_logits:
            # Keep branch logits for the shared loss kernel but release the
            # materialized final output at the call site when desired.
            output_logits = output_logits.detach() * 0.0
        return ReadState(
            h,
            hs,
            z0,
            zs,
            copy_read.log_attention,
            log_pcopy,
            copy_read.raw_gate,
            copy_read.g,
            evidence,
            pi_sem,
            pi_base,
            pi_sem,
            pi_copy,
            log_p0,
            log_ps,
            log_p,
            gauge,
            output_logits,
            diagnostics,
        )

    @staticmethod
    def _linear_chunk(lm_head: nn.Module, hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
        if not hasattr(lm_head, "weight"):
            return lm_head(hidden)[..., start:end].float()
        weight = lm_head.weight[start:end]
        bias = getattr(lm_head, "bias", None)
        if bias is not None:
            bias = bias[start:end]
        return torch.nn.functional.linear(hidden.float(), weight.float(), None if bias is None else bias.float())

    def loss_sum_from_hidden(
        self,
        h: torch.Tensor,
        hs: torch.Tensor,
        semantic_diagnostics: dict[str, torch.Tensor],
        copy_read: CopyRead,
        copy_state: CopyState,
        lm_head: nn.Module,
        labels: torch.Tensor,
        chunk_size: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the same mixture NLL while streaming vocabulary chunks.

        The target-token log probabilities are gathered from each chunk; no
        ``[B,T,V]`` final mixture tensor is materialized.  Linear heads use two
        passes (log-partition then target gather), while custom heads fall back
        to the dense oracle so the numerical contract remains explicit.
        """

        if labels.shape != h.shape[:2]:
            raise ValueError("labels must match hidden [B,T]")
        if not hasattr(lm_head, "weight"):
            dense = self(h, hs, semantic_diagnostics, copy_read, copy_state, lm_head, return_logits=True)
            return mixture_nll_from_state(dense, labels)
        safe_labels = labels.clamp_min(0).long()
        vocab_size = int(lm_head.weight.shape[0])
        chunk_size = max(1, int(chunk_size))
        flat_h, flat_hs = h.float().reshape(-1, h.shape[-1]), hs.float().reshape(-1, hs.shape[-1])
        log_z0 = torch.full((flat_h.shape[0], 1), -float("inf"), device=h.device)
        log_zs = torch.full_like(log_z0, -float("inf"))
        for start in range(0, vocab_size, chunk_size):
            end = min(vocab_size, start + chunk_size)
            log_z0 = torch.logaddexp(
                log_z0, self._linear_chunk(lm_head, flat_h, start, end).logsumexp(-1, keepdim=True)
            )
            log_zs = torch.logaddexp(
                log_zs, self._linear_chunk(lm_head, flat_hs, start, end).logsumexp(-1, keepdim=True)
            )
        target_l0 = torch.full((flat_h.shape[0],), torch.finfo(torch.float32).min, device=h.device)
        target_ls = torch.full_like(target_l0, torch.finfo(torch.float32).min)
        flat_targets = safe_labels.reshape(-1)
        for start in range(0, vocab_size, chunk_size):
            end = min(vocab_size, start + chunk_size)
            in_chunk = (flat_targets >= start) & (flat_targets < end)
            if not in_chunk.any():
                continue
            positions = in_chunk.nonzero(as_tuple=True)[0]
            z0 = self._linear_chunk(lm_head, flat_h.index_select(0, positions), start, end)
            zs = self._linear_chunk(lm_head, flat_hs.index_select(0, positions), start, end)
            local = flat_targets.index_select(0, positions) - start
            target_l0.index_copy_(
                0,
                positions,
                z0.gather(-1, local[:, None]).squeeze(-1) - log_z0.index_select(0, positions).squeeze(-1),
            )
            target_ls.index_copy_(
                0,
                positions,
                zs.gather(-1, local[:, None]).squeeze(-1) - log_zs.index_select(0, positions).squeeze(-1),
            )
        pi_base, pi_sem, pi_copy, _, _ = self._routing(h, semantic_diagnostics, copy_read, copy_state)
        copy_target = copy_target_log_probability(copy_read, copy_state, labels).reshape(-1)
        return _mixture_nll_from_targets(
            pi_base,
            pi_sem,
            pi_copy,
            target_l0.reshape_as(labels),
            target_ls.reshape_as(labels),
            copy_target.reshape_as(labels),
            labels,
        )

    @staticmethod
    def loss_sum_from_state(state: ReadState, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared dense/chunked NLL entry point used by the decoder and tests."""

        return mixture_nll_from_state(state, labels)

    def hidden_interpolation_state(
        self,
        h: torch.Tensor,
        delta: torch.Tensor,
        copy_read: CopyRead,
        copy_state: CopyState,
        lm_head: nn.Module,
    ) -> ReadState:
        """Named hidden-fusion control using the same legacy copy path."""

        hs = h.float() + float(self.hidden_lambda) * delta.float()
        zeros = h.new_zeros((*h.shape[:2], self.rank))
        evidence = copy_state.mask.any(dim=-1)[:, None, None].expand(-1, h.shape[1], -1).to(dtype=h.dtype)
        # This control uses the same copy gate but makes the hidden-fused
        # vocabulary branch observable whenever a source candidate exists.
        return self.forward(
            h,
            hs.to(h.dtype),
            {"q": zeros, "u": zeros, "evidence": evidence},
            copy_read,
            copy_state,
            lm_head,
        )


__all__ = [
    "DualReadout",
    "ReadState",
    "mixture_nll_from_state",
    "normalized_copy_entropy",
    "target_log_probability",
]
