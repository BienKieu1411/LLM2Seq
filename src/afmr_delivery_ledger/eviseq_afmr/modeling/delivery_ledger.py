"""Causal accounting of source evidence already delivered to the summary."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(initial: float, maximum: float) -> float:
    if not 0.0 < initial < maximum:
        raise ValueError("ledger strengths must satisfy 0 < initial < maximum")
    return math.log(initial / (maximum - initial))


def _atanh_ratio(initial: float, maximum: float) -> float:
    if not 0.0 < initial < maximum:
        raise ValueError("ledger strengths must satisfy 0 < initial < maximum")
    return math.atanh(initial / maximum)


def pool_source_regions(
    memory: torch.Tensor, content_mask: torch.Tensor, width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool contiguous content tokens and retain their exact region membership."""
    if width <= 0:
        raise ValueError("delivery-ledger region width must be positive")
    if memory.ndim != 3 or content_mask.shape != memory.shape[:2]:
        raise ValueError("memory and content mask must have matching source dimensions")
    batch, source_length, hidden = memory.shape
    valid = content_mask.bool()
    ordinal = valid.long().cumsum(dim=1) - 1
    source_region_ids = torch.where(valid, ordinal.div(width, rounding_mode="floor"), -1)
    region_count = max(1, (source_length + width - 1) // width)
    destination = source_region_ids.clamp_min(0)
    counts = memory.new_zeros(batch, region_count, dtype=torch.float32).scatter_add(1, destination, valid.float())
    regions = memory.new_zeros(batch, region_count, hidden, dtype=torch.float32).scatter_add(
        1,
        destination[..., None].expand(-1, -1, hidden),
        memory.float() * valid[..., None],
    )
    regions = regions / counts.clamp_min(1.0)[..., None]
    return regions.to(memory.dtype), counts.gt(0), source_region_ids


class SourceDeliveryLedger(nn.Module):
    """Track which coarse source regions have already supported generated tokens."""

    def __init__(self, hidden_size: int, config: dict):
        super().__init__()
        rank = int(config.get("rank", 128))
        if rank <= 0:
            raise ValueError("delivery_ledger.rank must be positive")
        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.key = nn.Linear(hidden_size, rank, bias=False)
        self.write_gate = nn.Linear(2 * hidden_size, 1)
        self.write_max = float(config.get("write_strength_max", 0.25))
        self.write_top_k = int(config.get("write_top_k", 4))
        if self.write_top_k <= 0:
            raise ValueError("delivery_ledger.write_top_k must be positive")
        write_init = float(config.get("write_strength_init", 0.05))
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, _logit(write_init, self.write_max))

        self.coverage_max = float(config.get("coverage_strength_max", 3.0))
        self.coverage_raw = nn.Parameter(
            torch.tensor(
                _logit(float(config.get("coverage_strength_init", 1.5)), self.coverage_max),
                dtype=torch.float32,
            )
        )
        self.read_enabled = bool(config.get("read_enabled", False))
        if self.read_enabled:
            self.read_max = float(config.get("read_strength_max", 0.20))
            self.read_raw = nn.Parameter(
                torch.tensor(
                    _logit(float(config.get("read_strength_init", 0.05)), self.read_max),
                    dtype=torch.float32,
                )
            )
        self.copy_max = float(config.get("copy_strength_max", 1.0))
        self.copy_gate = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(
            self.copy_gate.bias,
            _atanh_ratio(float(config.get("copy_strength_init", 0.50)), self.copy_max),
        )
        self.cross_max = float(config.get("cross_strength_max", 1.0))
        self.cross_gate = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.cross_gate.weight)
        nn.init.constant_(
            self.cross_gate.bias,
            _atanh_ratio(float(config.get("cross_strength_init", 0.50)), self.cross_max),
        )
        self.rank = rank
        self._coverage: torch.Tensor | None = None
        self._cross_coverage: torch.Tensor | None = None

    @staticmethod
    def _norm(states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(states.float(), (states.shape[-1],))

    def forward(
        self,
        hidden: torch.Tensor,
        regions: torch.Tensor,
        region_mask: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        coverage: torch.Tensor | None = None,
        relevance_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden.ndim != 3 or regions.ndim != 3:
            raise ValueError("ledger hidden states and source regions must be rank three")
        if hidden.shape[0] != regions.shape[0] or region_mask.shape != regions.shape[:2]:
            raise ValueError("ledger batch and region dimensions disagree")
        batch, target_length, _ = hidden.shape
        if active_mask is None:
            active_mask = torch.ones(batch, target_length, dtype=torch.bool, device=hidden.device)
        if active_mask.shape != hidden.shape[:2]:
            raise ValueError("ledger active mask must match decoder time steps")
        if not bool(region_mask.bool().any(-1).all()):
            raise ValueError("every example requires at least one content region")

        if relevance_override is None:
            normalized_hidden = self._norm(hidden).to(self.query.weight.dtype)
            normalized_regions = self._norm(regions).to(self.key.weight.dtype)
            queries = self.query(normalized_hidden).float()
            keys = self.key(normalized_regions).float()
            relevance = torch.matmul(queries, keys.transpose(1, 2)).float() / math.sqrt(self.rank)
        else:
            if relevance_override.shape != (batch, target_length, regions.shape[1]):
                raise ValueError("ledger relevance override must match decoder steps and source regions")
            relevance = relevance_override.float()
        if coverage is None:
            coverage = hidden.new_zeros(batch, regions.shape[1], dtype=torch.float32)
        elif coverage.shape != region_mask.shape:
            raise ValueError("ledger coverage must match source regions")
        else:
            coverage = coverage.float()

        valid_regions = region_mask[:, None, :].bool()
        relevance = relevance.masked_fill(~valid_regions, torch.finfo(torch.float32).min)

        # Only a few regions receive exposure at each step; a dense write
        # would mark every source region as covered at nearly the same rate.
        top_indices = relevance.topk(min(self.write_top_k, regions.shape[1]), dim=-1).indices
        selected = torch.zeros_like(valid_regions.expand_as(relevance)).scatter(-1, top_indices, True)
        exposure = torch.softmax(relevance.masked_fill(~(selected & valid_regions), -torch.inf), dim=-1)
        exposed_read = torch.matmul(exposure, regions.float()).float()
        gate_input = torch.cat((hidden.float(), exposed_read), dim=-1).to(self.write_gate.weight.dtype)
        write = self.write_max * torch.sigmoid(self.write_gate(gate_input).float())
        increment = write * exposure * active_mask[..., None].float()

        # The exclusive cumulative sum is exactly the causal exposure before
        # predicting each token. It matches the cached one-step update below.
        cumulative = increment.cumsum(dim=1)
        before = cumulative - increment
        coverage_before = 1.0 - (1.0 - coverage[:, None, :]) * torch.exp(-before)
        coverage_after = 1.0 - (1.0 - coverage) * torch.exp(-cumulative[:, -1])

        coverage_strength = self.coverage_max * torch.sigmoid(self.coverage_raw.float())
        normalized_for_gate = self._norm(hidden)
        copy_strength = self.copy_max * torch.tanh(
            self.copy_gate(normalized_for_gate.to(self.copy_gate.weight.dtype)).float()
        )
        cross_strength = self.cross_max * torch.tanh(
            self.cross_gate(normalized_for_gate.to(self.cross_gate.weight.dtype)).float()
        )
        routed = hidden
        if self.read_enabled:
            base_probability = torch.softmax(relevance, dim=-1)
            novelty = torch.softmax(relevance - coverage_strength * coverage_before, dim=-1)
            read_delta = torch.matmul(novelty - base_probability, regions.float()).float()
            hidden_float = hidden.float()
            hidden_rms = hidden_float.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            delta_rms = (read_delta.square().mean(-1, keepdim=True) + 1e-12).sqrt()
            read_strength = self.read_max * torch.sigmoid(self.read_raw.float())
            fused = hidden_float + read_strength * hidden_rms * read_delta / torch.maximum(delta_rms, 0.05 * hidden_rms)
            fused = fused * (hidden_rms / fused.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6))
            routed = torch.where(
                active_mask[..., None] & read_delta.abs().amax(-1, keepdim=True).gt(1e-7), fused, hidden_float
            ).to(hidden.dtype)

        valid_float = valid_regions.float()
        mean_coverage = (coverage_before * valid_float).sum(-1, keepdim=True) / valid_float.sum(-1, keepdim=True)
        copy_prior = -copy_strength * coverage_strength * (coverage_before - mean_coverage)
        copy_prior = torch.where(active_mask[..., None] & valid_regions, copy_prior, 0.0)
        cross_prior = -cross_strength * coverage_strength * (coverage_before - mean_coverage)
        cross_prior = torch.where(active_mask[..., None] & valid_regions, cross_prior, 0.0)
        return routed, copy_prior, cross_prior, coverage_after

    def get_coverage(self) -> torch.Tensor | None:
        return self._coverage

    def set_coverage(self, coverage: torch.Tensor) -> None:
        self._coverage = coverage

    def get_cross_coverage(self) -> torch.Tensor | None:
        return self._cross_coverage

    def set_cross_coverage(self, coverage: torch.Tensor) -> None:
        self._cross_coverage = coverage

    def clear_state(self) -> None:
        self._coverage = None
        self._cross_coverage = None

    def select_state(self, indices: torch.Tensor) -> None:
        if self._coverage is not None:
            self._coverage = self._coverage.index_select(0, indices)
        if self._cross_coverage is not None:
            self._cross_coverage = self._cross_coverage.index_select(0, indices)
