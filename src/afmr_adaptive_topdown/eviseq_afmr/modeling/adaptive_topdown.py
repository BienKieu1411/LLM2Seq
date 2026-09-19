"""Controller-conditioned regional pooling with a bounded top-down key correction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_bounded_init(value: float, maximum: float) -> float:
    if not 0.0 < float(value) < float(maximum):
        raise ValueError("bounded gate requires 0 < init < maximum")
    return math.log(float(value) / (float(maximum) - float(value)))


class _RegionMixerLayer(nn.Module):
    """Small pre-norm Transformer block over the compact region sequence."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attention_norm = nn.RMSNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)
        self.mlp_norm = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            nn.SiLU(),
            nn.Linear(4 * dim, dim, bias=False),
        )

    def forward(self, regions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # MultiheadAttention cannot consume an all-masked row. A zero sentinel
        # keeps that row finite and is removed again before returning.
        safe_valid = valid.clone()
        no_valid = ~safe_valid.any(dim=-1)
        safe_valid[no_valid, 0] = True
        safe_regions = regions.masked_fill(~valid.unsqueeze(-1), 0.0)
        normalized = self.attention_norm(safe_regions)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        regions = (safe_regions + attended).masked_fill(~valid.unsqueeze(-1), 0.0)
        regions = (regions + self.mlp(self.mlp_norm(regions))).masked_fill(~valid.unsqueeze(-1), 0.0)
        return regions


class AdaptiveTopDownKeyBridge(nn.Module):
    """Build a compact document view and broadcast it back to token keys.

    Regional pooling is adaptive but efficient: each content token receives a
    controller-conditioned positive weight once, and prefix sums compute the
    normalized weighted mean for every overlapping region. This avoids a
    ``[batch, regions, width, hidden]`` activation at long source lengths.
    """

    def __init__(self, memory_hidden: int, controller_dim: int, settings: dict):
        super().__init__()
        self.memory_hidden = int(memory_hidden)
        self.dim = int(settings["dim"])
        self.region_width = int(settings["region_width"])
        self.region_stride = int(settings["region_stride"])
        self.gate_max = float(settings["gate_max"])
        self.max_relative_rms = float(settings["max_relative_rms"])

        self.token_norm = nn.RMSNorm(self.memory_hidden)
        self.token_features = nn.Linear(self.memory_hidden, self.dim, bias=False)
        self.pool_controller = nn.Linear(controller_dim, self.dim, bias=False)
        self.pool_score = nn.Linear(self.dim, 1, bias=False)
        self.region_values = nn.Linear(self.memory_hidden, self.dim, bias=False)
        self.region_position = nn.Linear(4, self.dim, bias=False)
        self.region_controller = nn.Linear(controller_dim, self.dim, bias=False)
        self.region_norm = nn.RMSNorm(self.dim)
        self.mixers = nn.ModuleList(
            _RegionMixerLayer(self.dim, int(settings["num_heads"])) for _ in range(int(settings["mixer_layers"]))
        )

        self.token_query = nn.Linear(self.dim, self.dim, bias=False)
        self.topdown_attention = nn.MultiheadAttention(
            self.dim,
            int(settings["num_heads"]),
            dropout=0.0,
            batch_first=True,
        )
        topdown_rank = int(settings["topdown_rank"])
        self.topdown_norm = nn.RMSNorm(3 * self.dim)
        self.topdown_down = nn.Linear(3 * self.dim, topdown_rank, bias=False)
        self.topdown_out = nn.Linear(topdown_rank, self.memory_hidden, bias=False)
        nn.init.normal_(
            self.topdown_out.weight,
            mean=0.0,
            std=float(settings["output_init_rms"]) / math.sqrt(topdown_rank),
        )
        self.gate_raw = nn.Linear(controller_dim, 1)
        nn.init.zeros_(self.gate_raw.weight)
        nn.init.constant_(self.gate_raw.bias, _inverse_bounded_init(settings["gate_init"], self.gate_max))

    @staticmethod
    def _compact_content(values: torch.Tensor, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, hidden = values.shape
        content = content.bool()
        compact_index = (content.long().cumsum(dim=-1) - 1).clamp_min(0)
        compact = values.new_zeros(batch, length, hidden).scatter_add(
            1,
            compact_index.unsqueeze(-1).expand(-1, -1, hidden),
            values * content.unsqueeze(-1).to(values.dtype),
        )
        count = content.sum(dim=-1)
        valid = torch.arange(length, device=values.device).unsqueeze(0) < count.unsqueeze(1)
        return compact, valid

    @staticmethod
    def _gather_prefix(prefix: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return prefix.gather(1, indices.unsqueeze(-1).expand(-1, -1, prefix.shape[-1]))

    def _pool_regions(
        self,
        memory: torch.Tensor,
        content: torch.Tensor,
        controller: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = memory.shape
        normalized = self.token_norm(memory.float())
        token_features = self.token_features(normalized)
        region_values = self.region_values(normalized)
        compact_features, compact_valid = self._compact_content(token_features, content)
        compact_values, _ = self._compact_content(region_values, content)
        content_count = compact_valid.sum(dim=-1)

        token_scores = self.pool_score(
            F.silu(compact_features + self.pool_controller(controller.float()).unsqueeze(1))
        ).squeeze(-1)
        # Prefix-sum pooling subtracts two cumulative sums.  Bound the
        # learned pooling logits before exponentiation so a long document
        # cannot create an ill-conditioned ``prefix[end] - prefix[start]``
        # through a single extreme token.  This keeps the adaptive weights
        # positive while preserving useful within-window contrast.
        token_scores = 6.0 * torch.tanh(token_scores / 6.0)
        # One row-wise shift is shared by all windows and therefore cancels in
        # every regional softmax. It prevents overflow without changing weights.
        masked_scores = token_scores.masked_fill(~compact_valid, torch.finfo(token_scores.dtype).min)
        row_max = masked_scores.max(dim=-1, keepdim=True).values
        row_max = torch.where(content_count.unsqueeze(-1) > 0, row_max, torch.zeros_like(row_max))
        shifted_scores = (token_scores - row_max).masked_fill(~compact_valid, torch.finfo(token_scores.dtype).min)
        positive_weights = torch.exp(shifted_scores)

        weighted_values = compact_values * positive_weights.unsqueeze(-1)
        value_prefix = F.pad(weighted_values.cumsum(dim=1), (0, 0, 1, 0))
        weight_prefix = F.pad(positive_weights.cumsum(dim=1), (1, 0))

        regular_1d = torch.arange(
            0,
            max(1, length - self.region_width + 1),
            self.region_stride,
            device=memory.device,
        )
        regular = regular_1d.unsqueeze(0).expand(batch, -1)
        last_start = (content_count - self.region_width).clamp_min(0).unsqueeze(-1)
        starts = torch.cat((regular, last_start), dim=1)
        ends = torch.minimum(starts + self.region_width, content_count.unsqueeze(-1))
        regular_valid = regular <= last_start
        last_valid = (last_start.remainder(self.region_stride) != 0) | ~regular_valid[:, :1]
        valid = torch.cat((regular_valid, last_valid), dim=1)
        valid = valid & (starts < content_count.unsqueeze(-1)) & (ends > starts)
        safe_starts = torch.minimum(starts, content_count.unsqueeze(-1))

        numerator = self._gather_prefix(value_prefix, ends) - self._gather_prefix(value_prefix, safe_starts)
        denominator = weight_prefix.gather(1, ends) - weight_prefix.gather(1, safe_starts)
        pooled = numerator / denominator.clamp_min(1.0e-8).unsqueeze(-1)

        count_scale = content_count.unsqueeze(-1).float().clamp_min(1.0)
        center = (safe_starts.float() + ends.float() - 1.0).mul(0.5)
        center = center / (content_count.unsqueeze(-1).float() - 1.0).clamp_min(1.0)
        span = (ends - safe_starts).float() / count_scale
        position = torch.stack(
            (center, span, torch.sin(math.pi * center), torch.cos(math.pi * center)),
            dim=-1,
        )
        regions = pooled + self.region_position(position) + self.region_controller(controller.float()).unsqueeze(1)
        regions = self.region_norm(regions).masked_fill(~valid.unsqueeze(-1), 0.0)
        for mixer in self.mixers:
            regions = mixer(regions, valid)
        return regions, valid

    def _bounded_correction(
        self,
        raw_delta: torch.Tensor,
        memory: torch.Tensor,
        controller: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.gate_max * torch.sigmoid(self.gate_raw(controller.float()))
        gated = raw_delta.float() * gate.unsqueeze(1)
        anchor_rms = torch.sqrt(memory.float().square().mean(dim=-1, keepdim=True) + 1.0e-12).detach()
        radius = self.max_relative_rms * anchor_rms
        delta_rms = torch.sqrt(gated.square().mean(dim=-1, keepdim=True) + 1.0e-12)
        return gated * radius / torch.sqrt(radius.square() + delta_rms.square() + 1.0e-12)

    def forward(self, memory: torch.Tensor, content_mask: torch.Tensor, controller: torch.Tensor) -> torch.Tensor:
        if memory.ndim != 3 or content_mask.shape != memory.shape[:2]:
            raise ValueError("adaptive top-down inputs must be memory [B,S,H] and mask [B,S]")
        content = content_mask.bool()
        regions, region_valid = self._pool_regions(memory.float(), content, controller.float())

        safe_valid = region_valid.clone()
        no_valid = ~safe_valid.any(dim=-1)
        safe_valid[no_valid, 0] = True
        safe_regions = regions.masked_fill(~region_valid.unsqueeze(-1), 0.0)
        token_features = self.token_features(self.token_norm(memory.float()))
        query = self.token_query(token_features)
        topdown, _ = self.topdown_attention(
            query,
            safe_regions,
            safe_regions,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        topdown = topdown.masked_fill(~content.unsqueeze(-1), 0.0)
        controller_features = self.pool_controller(controller.float()).unsqueeze(1).expand_as(topdown)
        fused = torch.cat((token_features, topdown, controller_features), dim=-1)
        raw_delta = self.topdown_out(F.silu(self.topdown_down(self.topdown_norm(fused))))
        correction = self._bounded_correction(raw_delta, memory, controller)
        return correction.masked_fill(~content.unsqueeze(-1), 0.0).to(memory.dtype)
