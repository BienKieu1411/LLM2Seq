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
        write_init = float(config.get("write_strength_init", 0.05))
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, _logit(write_init, self.write_max))

        self.coverage_max = float(config.get("coverage_strength_max", 1.0))
        self.coverage_raw = nn.Parameter(
            torch.tensor(
                _logit(float(config.get("coverage_strength_init", 0.10)), self.coverage_max),
                dtype=torch.float32,
            )
        )
        self.read_max = float(config.get("read_strength_max", 0.20))
        self.read_raw = nn.Parameter(
            torch.tensor(
                _logit(float(config.get("read_strength_init", 0.05)), self.read_max),
                dtype=torch.float32,
            )
        )
        self.copy_max = float(config.get("copy_strength_max", 0.50))
        self.copy_raw = nn.Parameter(
            torch.tensor(
                _logit(float(config.get("copy_strength_init", 0.10)), self.copy_max),
                dtype=torch.float32,
            )
        )
        self.rank = rank
        self._coverage: torch.Tensor | None = None

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        normalized_hidden = self._norm(hidden).to(self.query.weight.dtype)
        normalized_regions = self._norm(regions).to(self.key.weight.dtype)
        queries = self.query(normalized_hidden).float()
        keys = self.key(normalized_regions).float()
        if coverage is None:
            coverage = hidden.new_zeros(batch, regions.shape[1], dtype=torch.float32)
        elif coverage.shape != region_mask.shape:
            raise ValueError("ledger coverage must match source regions")
        else:
            coverage = coverage.float()

        coverage_strength = self.coverage_max * torch.sigmoid(self.coverage_raw.float())
        read_strength = self.read_max * torch.sigmoid(self.read_raw.float())
        copy_strength = self.copy_max * torch.sigmoid(self.copy_raw.float())
        floor = torch.finfo(torch.float32).min
        routed: list[torch.Tensor] = []
        copy_priors: list[torch.Tensor] = []
        valid_float = region_mask.float()
        valid_count = valid_float.sum(-1, keepdim=True).clamp_min(1.0)

        for index in range(target_length):
            active = active_mask[:, index : index + 1]
            relevance = torch.einsum("bd,bkd->bk", queries[:, index], keys) / math.sqrt(self.rank)
            scores = (relevance - coverage_strength * coverage).masked_fill(~region_mask, floor)
            probability = torch.softmax(scores, dim=-1)
            read = torch.einsum("bk,bkh->bh", probability, regions.float())
            hidden_step = hidden[:, index].float()
            hidden_rms = hidden_step.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            read_unit = self._norm(read)
            fused = hidden_step + read_strength * hidden_rms * read_unit
            fused = fused * (hidden_rms / fused.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6))
            routed.append(torch.where(active, fused, hidden_step).to(hidden.dtype))

            mean = (scores.masked_fill(~region_mask, 0.0) * valid_float).sum(-1, keepdim=True) / valid_count
            centered = (scores - mean).masked_fill(~region_mask, 0.0)
            copy_priors.append(torch.where(active, copy_strength * centered, torch.zeros_like(centered)))

            gate_input = torch.cat((hidden_step, read), dim=-1).to(self.write_gate.weight.dtype)
            write = self.write_max * torch.sigmoid(self.write_gate(gate_input).float())
            increment = write * probability * (1.0 - coverage)
            coverage = torch.where(active, coverage + increment, coverage)

        return torch.stack(routed, dim=1), torch.stack(copy_priors, dim=1), coverage

    def get_coverage(self) -> torch.Tensor | None:
        return self._coverage

    def set_coverage(self, coverage: torch.Tensor) -> None:
        self._coverage = coverage

    def clear_state(self) -> None:
        self._coverage = None

    def select_state(self, indices: torch.Tensor) -> None:
        if self._coverage is not None:
            self._coverage = self._coverage.index_select(0, indices)
