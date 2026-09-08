"""Causal source-region usage estimates, trained through the gold-token CE only."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrefixHistory:
    coverage: torch.Tensor
    recent: torch.Tensor

    def index_select(self, indices):
        return PrefixHistory(self.coverage.index_select(0, indices), self.recent.index_select(0, indices))


@dataclass
class ReadPlan:
    coverage: torch.Tensor
    recent: torch.Tensor

    def slice(self, start, end=None):
        return ReadPlan(self.coverage[:, start:end], self.recent[:, start:end])


class CoveragePlanner(nn.Module):
    """Track the observed summary prefix, not the next-token target or a draft.

    The tracking attention does not depend on coverage. A cumulative sum can
    therefore compute the entire teacher-forced prefix state in parallel.
    This is a learned usage estimate, not verified semantic fact coverage.
    """

    def __init__(self, hidden_size, rank, config):
        super().__init__()
        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.usage_gate = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.usage_gate.weight)
        nn.init.zeros_(self.usage_gate.bias)
        self.coverage_scale = float(config.get("coverage_scale", 8.0))
        if not 0 < self.coverage_scale < math.inf:
            raise ValueError("Planner coverage_scale must be finite and positive")
        self.use_coverage = bool(config.get("use_coverage", True))
        self.use_continuity = bool(config.get("use_continuity", True))
        self.coverage_max = float(config.get("coverage_max", 2.0))
        self.continuity_max = float(config.get("continuity_max", 2.0))
        self.coverage_raw = self._raw(config.get("coverage_init", 0.2), self.coverage_max)
        self.continuity_raw = self._raw(config.get("continuity_init", 0.2), self.continuity_max)

    @staticmethod
    def _raw(initial, maximum):
        if not 0 < float(initial) < float(maximum) < math.inf:
            raise ValueError("Planner strengths require 0 < init < max < infinity")
        return nn.Parameter(torch.tensor(math.log(initial / (maximum - initial)), dtype=torch.float32))

    @staticmethod
    def masked_softmax(scores, valid):
        floor = torch.finfo(torch.float32).min
        return scores.float().masked_fill(~valid, floor).softmax(-1).masked_fill(~valid, 0.0)

    def forward(self, hidden, source, summary_input_mask, history=None):
        if summary_input_mask.shape != hidden.shape[:2]:
            raise ValueError("Summary input mask must match the current decoder positions")
        batch, length, _ = hidden.shape
        regions = source.region_keys.shape[1]
        if history is None:
            history = PrefixHistory(hidden.new_zeros(batch, regions).float(), hidden.new_zeros(batch, regions).float())
        if history.coverage.shape != (batch, regions) or history.recent.shape != (batch, regions):
            raise ValueError("Prefix history and source-region cache disagree")
        normalized = F.rms_norm(hidden.float(), (hidden.shape[-1],))
        query = self.query(normalized.to(self.query.weight.dtype)).float()
        scores = (query @ source.region_keys.float().transpose(1, 2)).float() / math.sqrt(query.shape[-1])
        probabilities = self.masked_softmax(scores + source.region_bias[:, None, :], source.region_mask[:, None, :])
        mass = torch.sigmoid(self.usage_gate(normalized.to(self.usage_gate.weight.dtype)).float())
        usage = probabilities * mass * summary_input_mask[..., None]
        coverage = history.coverage[:, None, :] + usage.cumsum(1)
        # Forward-fill the latest valid summary input; prompt/padding adds no
        # usage and must not replace a cached continuation distribution.
        positions = torch.arange(1, length + 1, device=hidden.device).expand(batch, -1)
        last = positions.masked_fill(~summary_input_mask.bool(), 0).cummax(1).values
        bank = torch.cat((history.recent[:, None, :], probabilities), dim=1)
        recent = bank.gather(1, last[..., None].expand(-1, -1, regions))
        plan = ReadPlan(coverage, recent)
        final = PrefixHistory(coverage[:, -1], recent[:, -1]) if length else history
        return plan, final

    def bias(self, plan):
        coverage = self.coverage_max * self.coverage_raw.sigmoid()
        continuity = self.continuity_max * self.continuity_raw.sigmoid()
        return (
            -float(self.use_coverage) * coverage * torch.log1p(plan.coverage / self.coverage_scale)
            + float(self.use_continuity) * continuity * plan.recent
        )
