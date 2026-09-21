from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class CopyState:
    keys: torch.Tensor
    token_ids: torch.Tensor
    mask: torch.Tensor
    bias: torch.Tensor
    region_ids: torch.Tensor | None = None

    def index_select(self, indices: torch.Tensor) -> CopyState:
        return CopyState(
            *(value.index_select(0, indices) for value in (self.keys, self.token_ids, self.mask, self.bias)),
            None if self.region_ids is None else self.region_ids.index_select(0, indices),
        )


class GroundedCopyHead(nn.Module):
    def __init__(
        self, hidden_size: int, key_dim: int, gate_init: float, route_init: float = 0.0, route_max: float = 0.5
    ):
        super().__init__()
        if key_dim <= 0 or not 0 < gate_init < 1:
            raise ValueError("Grounded copy requires key_dim > 0 and 0 < gate_init < 1")
        self.query = nn.Linear(hidden_size, key_dim, bias=False)
        self.context_key = nn.Linear(hidden_size, key_dim, bias=False)
        self.lexical_key = nn.Linear(hidden_size, key_dim, bias=False)
        self.gate = nn.Linear(2 * key_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(gate_init / (1 - gate_init)))
        if route_init:
            if not 0 < route_init < route_max:
                raise ValueError("copy route strength must satisfy 0 < init < max")
            self.route_strength_raw = nn.Parameter(torch.tensor(math.log(route_init / (route_max - route_init))))
            self.route_strength_max = float(route_max)
        else:
            self.register_parameter("route_strength_raw", None)
            self.route_strength_max = 0.0

    @staticmethod
    def _norm(states):
        return F.rms_norm(states.float(), (states.shape[-1],))

    def prepare(
        self,
        memory,
        source_bias,
        content_mask,
        embedding,
        *,
        copy_token_ids,
        copy_token_mask,
        copy_encoder_indices,
        copy_token_indices,
        copy_alignment_weights,
        source_region_ids: torch.Tensor | None = None,
    ) -> CopyState:
        batch, width = copy_token_ids.shape
        context = self.context_key(self._norm(memory).to(self.context_key.weight.dtype))
        rank = context.shape[-1]
        valid = content_mask.gather(1, copy_encoder_indices).float()
        weights = copy_alignment_weights.float() * valid
        destination = copy_token_indices[..., None].expand(-1, -1, rank)
        pooled = (
            context.new_zeros(batch, width, rank)
            .float()
            .scatter_add(
                1,
                destination,
                context.gather(1, copy_encoder_indices[..., None].expand(-1, -1, rank)).float() * weights[..., None],
            )
        )
        totals = weights.new_zeros(batch, width).scatter_add(1, copy_token_indices, weights)
        pooled = pooled / totals.clamp_min(1e-8)[..., None]
        bias = source_bias.new_zeros(batch, width).float().scatter_add(
            1, copy_token_indices, source_bias.float().gather(1, copy_encoder_indices) * weights
        ) / totals.clamp_min(1e-8)
        unique_ids, inverse = torch.unique(copy_token_ids, return_inverse=True)
        lexical_bank = self.lexical_key(self._norm(embedding(unique_ids)).to(self.lexical_key.weight.dtype))
        lexical = lexical_bank[inverse]
        keys = self._norm(pooled + lexical.float())
        region_ids = None
        if source_region_ids is not None:
            aligned_regions = source_region_ids.gather(1, copy_encoder_indices)
            best_weight = weights.new_zeros(batch, width).scatter_reduce(
                1, copy_token_indices, weights, reduce="amax", include_self=True
            )
            candidate = torch.where(
                weights.gt(0) & weights.eq(best_weight.gather(1, copy_token_indices)),
                aligned_regions,
                source_region_ids.shape[1],
            )
            region_ids = torch.full_like(copy_token_ids, source_region_ids.shape[1]).scatter_reduce(
                1, copy_token_indices, candidate, reduce="amin", include_self=True
            )
            region_ids = torch.where(copy_token_mask & totals.gt(0), region_ids, -1)
            valid_copy = copy_token_mask & totals.gt(0)
            if bool(((region_ids < 0) | (region_ids >= source_region_ids.shape[1]))[valid_copy].any()):
                raise ValueError("valid copy tokens require a source region")
        return CopyState(keys, copy_token_ids, copy_token_mask & totals.gt(0), bias, region_ids)

    def distribution(self, hidden: torch.Tensor, state: CopyState, route_scores: torch.Tensor | None = None):
        query = self.query(self._norm(hidden).to(self.query.weight.dtype)).float()
        scores = torch.matmul(query, state.keys.float().transpose(1, 2)) / math.sqrt(query.shape[-1])
        scores = scores + state.bias.float()[:, None, :]
        if route_scores is not None:
            if state.region_ids is None or self.route_strength_raw is None:
                raise ValueError("copy route scores require enabled routing and mapped source regions")
            if route_scores.shape[:2] != hidden.shape[:2]:
                raise ValueError("copy route scores must match decoder time steps")
            if bool((state.region_ids[state.mask] >= route_scores.shape[-1]).any()):
                raise ValueError("copy region ID exceeds routed region count")
            selected = route_scores.float().gather(
                -1, state.region_ids.clamp_min(0)[:, None, :].expand(-1, hidden.shape[1], -1)
            )
            strength = self.route_strength_max * torch.sigmoid(self.route_strength_raw.float())
            scores = scores + strength * selected
        floor = torch.finfo(torch.float32).min
        has_source = state.mask.any(-1)[:, None, None]
        log_attention = F.log_softmax(scores.masked_fill(~state.mask[:, None, :], floor), dim=-1)
        log_attention = log_attention.masked_fill(~state.mask[:, None, :], floor)
        context = torch.matmul(log_attention.exp(), state.keys.float())
        gate = self.gate(torch.cat((query, context), dim=-1).to(self.gate.weight.dtype)).float()
        log_copy = torch.where(has_source, F.logsigmoid(gate), floor)
        log_generate = torch.where(has_source, F.logsigmoid(-gate), 0.0)
        return log_attention, log_copy, log_generate

    def mix_logits(
        self, hidden: torch.Tensor, logits: torch.Tensor, state: CopyState, route_scores: torch.Tensor | None = None
    ) -> torch.Tensor:
        log_attention, log_copy, log_generate = self.distribution(hidden, state, route_scores)
        source_probability = logits.new_zeros(logits.shape, dtype=torch.float32).scatter_add(
            -1, state.token_ids[:, None, :].expand(-1, hidden.shape[1], -1), log_attention.exp()
        )
        normalizer = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        log_source = source_probability.clamp_min(torch.finfo(torch.float32).tiny).log()
        log_source = log_source.masked_fill(source_probability.eq(0), torch.finfo(torch.float32).min)
        mixed = torch.logaddexp(logits.float() - normalizer + log_generate, log_source + log_copy) + normalizer
        return torch.where(state.mask.any(-1)[:, None, None], mixed, logits.float())

    def loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        state: CopyState,
        lm_head,
        chunk_size: int,
        route_scores: torch.Tensor | None = None,
    ):
        def chunk_loss(states, targets, route):
            log_attention, log_copy, log_generate = self.distribution(states, state, route)
            matches = targets[..., None].eq(state.token_ids[:, None, :]) & state.mask[:, None, :]
            copy_target = torch.logsumexp(log_attention.masked_fill(~matches, torch.finfo(torch.float32).min), dim=-1)
            valid = targets.ne(-100)
            lm_logits = lm_head(states[valid]).float()
            lm_target = -F.cross_entropy(lm_logits, targets[valid], reduction="none")
            return -torch.logaddexp(
                lm_target + log_generate[..., 0][valid], copy_target[valid] + log_copy[..., 0][valid]
            ).sum()

        losses = []
        stride = max(1, chunk_size // hidden.shape[0])
        for start in range(0, hidden.shape[1], stride):
            states, targets = hidden[:, start : start + stride], labels[:, start : start + stride]
            route = None if route_scores is None else route_scores[:, start : start + stride]
            losses.append(
                checkpoint(chunk_loss, states, targets, route, use_reentrant=False)
                if torch.is_grad_enabled()
                else chunk_loss(states, targets, route)
            )
        total = torch.stack(losses).sum() if losses else hidden.sum() * 0.0
        return total / labels.ne(-100).sum().clamp_min(1)
