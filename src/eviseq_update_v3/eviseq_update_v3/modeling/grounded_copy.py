from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .semantic_plan import CoveragePlanner, ReadPlan


@dataclass
class CopyState:
    keys: torch.Tensor
    token_ids: torch.Tensor
    mask: torch.Tensor
    bias: torch.Tensor
    semantic_values: torch.Tensor | None = None
    semantic_keys: torch.Tensor | None = None
    semantic_mask: torch.Tensor | None = None
    semantic_bias: torch.Tensor | None = None

    def index_select(self, indices: torch.Tensor) -> CopyState:
        return type(self)(
            **{
                field.name: None if (value := getattr(self, field.name)) is None else value.index_select(0, indices)
                for field in fields(self)
            }
        )


@dataclass
class PlannedCopyState(CopyState):
    token_regions: torch.Tensor | None = None
    region_keys: torch.Tensor | None = None
    region_mask: torch.Tensor | None = None
    region_bias: torch.Tensor | None = None
    prompt_lengths: torch.Tensor | None = None


class GroundedCopyHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        key_dim: int,
        gate_init: float,
        *,
        semantic_read: bool = False,
        semantic_rank: int = 128,
        semantic_gate_init: float = 0.05,
        semantic_attention: str = "shared_copy",
        semantic_max_relative_rms: float | None = None,
        semantic_num_heads: int = 1,
        semantic_fusion: str = "residual",
        semantic_head_gate_position: str = "pre_norm",
        semantic_planner: dict | None = None,
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
        if semantic_rank <= 0 or not 0 < semantic_gate_init < 1:
            raise ValueError("Semantic read requires rank > 0 and 0 < gate_init < 1")
        if semantic_attention not in {"shared_copy", "independent_source", "hierarchical_coverage"}:
            raise ValueError("Unknown semantic attention mode")
        if type(semantic_num_heads) is not int or semantic_num_heads < 1 or semantic_rank % semantic_num_heads:
            raise ValueError("semantic_num_heads must be a positive integer dividing semantic_rank")
        if semantic_num_heads > 1 and semantic_attention == "shared_copy":
            raise ValueError("Multiple semantic heads require independent_source or hierarchical_coverage attention")
        if semantic_max_relative_rms is not None and not 0 < float(semantic_max_relative_rms) < math.inf:
            raise ValueError("semantic_max_relative_rms must be null or a finite positive number")
        if semantic_fusion not in {"residual", "norm_preserving"}:
            raise ValueError("semantic_fusion must be residual or norm_preserving")
        if semantic_head_gate_position not in {"pre_norm", "post_norm"}:
            raise ValueError("semantic_head_gate_position must be pre_norm or post_norm")
        if semantic_fusion == "norm_preserving" and (
            semantic_max_relative_rms is None or not 0 < float(semantic_max_relative_rms) < 1
        ):
            raise ValueError("Norm-preserving fusion requires 0 < max_relative_rms < 1")
        self.semantic_read_enabled = bool(semantic_read)
        self.semantic_attention = semantic_attention
        self.semantic_fusion = semantic_fusion
        # Missing fields retain the old graph for resolved configs/checkpoints.
        # Shipped recipes explicitly select post_norm and unrestricted heads.
        self.semantic_head_gate_position = semantic_head_gate_position
        self.semantic_num_heads = semantic_num_heads
        self.semantic_head_dim = semantic_rank // semantic_num_heads
        self.semantic_max_relative_rms = None if semantic_max_relative_rms is None else float(semantic_max_relative_rms)
        self.semantic_value = self.semantic_output = self.semantic_gate = None
        self.semantic_key = self.semantic_query = None
        self.planner = None
        self.semantic_head_gate = None
        planner_config = semantic_planner or {}
        self.partition_heads = bool(planner_config.get("partition_heads", True))
        self.region_size = int(planner_config.get("region_size", 64))
        if self.region_size < 1:
            raise ValueError("Planner region_size must be positive")
        if self.semantic_read_enabled:
            # Keep downstream initialization identical to the copy-only control.
            with torch.random.fork_rng(devices=[]):
                self.semantic_value = nn.Linear(hidden_size, semantic_rank, bias=False)
                self.semantic_output = nn.Linear(semantic_rank, hidden_size, bias=False)
                query_dim = semantic_rank if semantic_attention != "shared_copy" else key_dim
                self.semantic_gate = nn.Linear(query_dim + semantic_rank, 1)
                nn.init.zeros_(self.semantic_output.weight)
                nn.init.zeros_(self.semantic_gate.weight)
                nn.init.constant_(self.semantic_gate.bias, math.log(semantic_gate_init / (1 - semantic_gate_init)))
                if semantic_attention != "shared_copy":
                    self.semantic_key = nn.Linear(hidden_size, semantic_rank, bias=False)
                    self.semantic_query = nn.Linear(hidden_size, semantic_rank, bias=False)
                if semantic_attention == "hierarchical_coverage":
                    self.planner = CoveragePlanner(hidden_size, semantic_rank, planner_config)
                    self.semantic_head_gate = nn.Linear(hidden_size, semantic_num_heads)
                    nn.init.zeros_(self.semantic_head_gate.weight)
                    nn.init.zeros_(self.semantic_head_gate.bias)

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
    ) -> CopyState:
        batch, width = copy_token_ids.shape
        normalized_memory = self._norm(memory)
        context = self.context_key(normalized_memory.to(self.context_key.weight.dtype))
        valid = content_mask.gather(1, copy_encoder_indices).float()
        weights = copy_alignment_weights.float() * valid
        totals = weights.new_zeros(batch, width).scatter_add(1, copy_token_indices, weights)

        def overlap_pool(projected):
            rank = projected.shape[-1]
            pooled = (
                projected.new_zeros(batch, width, rank)
                .float()
                .scatter_add(
                    1,
                    copy_token_indices[..., None].expand(-1, -1, rank),
                    projected.gather(1, copy_encoder_indices[..., None].expand(-1, -1, rank)).float()
                    * weights[..., None],
                )
            )
            return pooled / totals.clamp_min(1e-8)[..., None]

        pooled = overlap_pool(context)
        bias = source_bias.new_zeros(batch, width).float().scatter_add(
            1, copy_token_indices, source_bias.float().gather(1, copy_encoder_indices) * weights
        ) / totals.clamp_min(1e-8)
        unique_ids, inverse = torch.unique(copy_token_ids, return_inverse=True)
        lexical_bank = self.lexical_key(self._norm(embedding(unique_ids)).to(self.lexical_key.weight.dtype))
        lexical = lexical_bank[inverse]
        keys = self._norm(pooled + lexical.float())
        semantic_values = None
        semantic_keys = semantic_mask = semantic_bias = None
        if self.semantic_read_enabled:
            semantic_values = self.semantic_value(normalized_memory.to(self.semantic_value.weight.dtype))
            if self.semantic_attention != "shared_copy":
                # Read native encoder positions. Copy-token boundaries and lexical
                # embeddings must not decide which semantic values the LM reads.
                projected_keys = self.semantic_key(normalized_memory.to(self.semantic_key.weight.dtype))
                # Each head has its own scale. Keep the flat cache layout and
                # total projection width independent of the number of heads.
                semantic_keys = self._norm(
                    projected_keys.reshape(batch, memory.shape[1], self.semantic_num_heads, self.semantic_head_dim)
                ).flatten(-2)
                semantic_mask = content_mask.bool()
                semantic_bias = source_bias.float()
            else:
                semantic_values = overlap_pool(semantic_values)
        state = CopyState(
            keys,
            copy_token_ids,
            copy_token_mask & totals.gt(0),
            bias,
            semantic_values,
            semantic_keys,
            semantic_mask,
            semantic_bias,
        )
        if self.planner is not None:
            count = max(1, (memory.shape[1] + self.region_size - 1) // self.region_size)
            region_ids = (content_mask.long().cumsum(-1) - 1).clamp_min(0) // self.region_size
            valid = content_mask.float()
            counts = valid.new_zeros(batch, count).scatter_add(1, region_ids, valid)
            rank = semantic_keys.shape[-1]
            region_keys = (
                semantic_keys.new_zeros(batch, count, rank).scatter_add(
                    1, region_ids[..., None].expand(-1, -1, rank), semantic_keys * valid[..., None]
                )
                / counts.clamp_min(1)[..., None]
            )
            region_bias = source_bias.new_zeros(batch, count).float().scatter_add(
                1, region_ids, source_bias.float() * valid
            ) / counts.clamp_min(1)
            state = PlannedCopyState(
                **vars(state),
                token_regions=region_ids,
                region_keys=self._norm(
                    region_keys.reshape(batch, count, self.semantic_num_heads, self.semantic_head_dim)
                ).flatten(-2),
                region_mask=counts.gt(0),
                region_bias=region_bias,
            )
        return state

    def _attention(self, hidden: torch.Tensor, state: CopyState):
        query = self.query(self._norm(hidden).to(self.query.weight.dtype)).float()
        scores = torch.matmul(query, state.keys.float().transpose(1, 2)) / math.sqrt(query.shape[-1])
        scores = scores + state.bias.float()[:, None, :]
        floor = torch.finfo(torch.float32).min
        has_source = state.mask.any(-1)[:, None, None]
        log_attention = F.log_softmax(scores.masked_fill(~state.mask[:, None, :], floor), dim=-1)
        log_attention = log_attention.masked_fill(~state.mask[:, None, :], floor)
        context = torch.matmul(log_attention.exp(), state.keys.float())
        gate = self.gate(torch.cat((query, context), dim=-1).to(self.gate.weight.dtype)).float()
        log_copy = torch.where(has_source, F.logsigmoid(gate), floor)
        log_generate = torch.where(has_source, F.logsigmoid(-gate), 0.0)
        return query, log_attention, log_copy, log_generate

    def distribution(self, hidden: torch.Tensor, state: CopyState):
        return self._attention(hidden, state)[1:]

    def _native_context(self, query: torch.Tensor, state: CopyState) -> torch.Tensor:
        if self.semantic_num_heads == 1:
            # Preserve the v2 arithmetic for the one-head ablation, including
            # exact zero attention on padding and fully masked documents.
            scores = torch.matmul(query, state.semantic_keys.float().transpose(1, 2)) / math.sqrt(query.shape[-1])
            scores = scores + state.semantic_bias.float()[:, None, :]
            floor = torch.finfo(torch.float32).min
            valid = state.semantic_mask[:, None, :]
            log_attention = F.log_softmax(scores.masked_fill(~valid, floor), dim=-1)
            log_attention = log_attention.masked_fill(~valid, floor)
            return torch.matmul(log_attention.exp(), state.semantic_values.float())

        batch, length, rank = query.shape
        source_length = state.semantic_keys.shape[1]
        heads, width = self.semantic_num_heads, self.semantic_head_dim
        q = query.reshape(batch, length, heads, width).transpose(1, 2)
        k = state.semantic_keys.float().reshape(batch, source_length, heads, width).transpose(1, 2)
        v = state.semantic_values.float().reshape(batch, source_length, heads, width).transpose(1, 2)
        bias = state.semantic_bias.float()[:, None, None, :].masked_fill(
            ~state.semantic_mask[:, None, None, :], float("-inf")
        )
        # No target state is cached. SDPA can use fused kernels under CUDA
        # autocast, avoiding materialized [B,H,T,S] attention probabilities.
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)
        return context.transpose(1, 2).reshape(batch, length, rank).float()

    def _hierarchical_context(self, query, state: PlannedCopyState, plan: ReadPlan):
        if plan is None:
            raise ValueError("Hierarchical semantic read requires a causal prefix plan")
        batch, length, rank = query.shape
        heads, width = self.semantic_num_heads, self.semantic_head_dim
        regions, source_length = state.region_keys.shape[1], state.semantic_keys.shape[1]
        q = query.reshape(batch, length, heads, width).transpose(1, 2)
        region_keys = state.region_keys.float().reshape(batch, regions, heads, width).transpose(1, 2)
        keys = state.semantic_keys.float().reshape(batch, source_length, heads, width).transpose(1, 2)
        values = state.semantic_values.float().reshape(batch, source_length, heads, width).transpose(1, 2)
        region_scores = (q @ region_keys.transpose(-1, -2)).float() / math.sqrt(width)
        valid_regions = state.region_mask[:, None, None, :]
        if self.partition_heads:
            # Disjoint interleaved region ownership. Every valid source region
            # remains reachable, but two heads cannot spend their mass on the
            # same region. Learned output gates can suppress irrelevant heads.
            owner = torch.arange(regions, device=query.device) % heads
            head = torch.arange(heads, device=query.device)
            valid_regions = valid_regions & owner[None, None, None, :].eq(head[None, :, None, None])
        region_probability = self.planner.masked_softmax(
            region_scores + state.region_bias[:, None, None, :] + self.planner.bias(plan)[:, None, :, :],
            valid_regions,
        )
        # Full rank matching within each disjoint region. Normalize within a
        # region before multiplying by its plan probability, so a longer region
        # does not win merely because it contains more tokens.
        scores = (q @ keys.transpose(-1, -2)).float() / math.sqrt(width)
        scores = scores + state.semantic_bias[:, None, None, :]
        valid = state.semantic_mask[:, None, None, :]
        region_ids = state.token_regions[:, None, None, :].expand(-1, heads, length, -1)
        floor = torch.finfo(torch.float32).min
        scores = scores.masked_fill(~valid, floor)
        maximum = scores.new_full(region_probability.shape, floor).scatter_reduce(
            -1, region_ids, scores, reduce="amax", include_self=True
        )
        weights = (scores - maximum.gather(-1, region_ids)).exp().masked_fill(~valid, 0.0)
        totals = weights.new_zeros(region_probability.shape).scatter_add(-1, region_ids, weights)
        weights = weights / totals.gather(-1, region_ids).clamp_min(1e-12)
        weights = weights * region_probability.gather(-1, region_ids)
        context = (weights @ values).float().transpose(1, 2).reshape(batch, length, rank)
        return context

    def _fuse(self, hidden, delta):
        base = hidden.float()
        if self.semantic_fusion == "norm_preserving":
            # Remove radial motion: the new branch changes content direction,
            # not the magnitude of the pretrained vocabulary input.
            squared_norm = base.square().sum(-1, keepdim=True)
            denominator = torch.where(squared_norm > 0, squared_norm, torch.ones_like(squared_norm))
            delta = delta - base * ((base * delta).sum(-1, keepdim=True) / denominator)
        if self.semantic_max_relative_rms is not None:
            cap = self.semantic_max_relative_rms * torch.linalg.vector_norm(base, dim=-1, keepdim=True)
            cap = cap / math.sqrt(hidden.shape[-1])
            delta = delta * cap / torch.sqrt(cap.square() + delta.square().mean(-1, keepdim=True) + 1e-12)
        fused = base + delta
        if self.semantic_fusion == "norm_preserving":
            norm = torch.linalg.vector_norm(base, dim=-1, keepdim=True)
            fused_norm = torch.linalg.vector_norm(fused, dim=-1, keepdim=True)
            denominator = torch.where(fused_norm > 0, fused_norm, torch.ones_like(fused_norm))
            fused = torch.where(norm > 0, fused * (norm / denominator), base)
        return fused.to(hidden.dtype)

    def read(self, hidden: torch.Tensor, state: CopyState, plan: ReadPlan | None = None):
        """Condition the LM while preserving the exact-copy distribution.

        The one-head setting retains the v2 native-source read.
        """
        query, log_attention, log_copy, log_generate = self._attention(hidden, state)
        generation_hidden = hidden
        if self.semantic_read_enabled:
            gain = None
            if state.semantic_values is None:
                raise ValueError("Semantic read is enabled but cached source values are missing")
            semantic_mask = state.mask
            if self.semantic_attention != "shared_copy":
                if state.semantic_keys is None or state.semantic_mask is None or state.semantic_bias is None:
                    raise ValueError("Independent semantic read requires cached native source keys, mask and bias")
                query = self.semantic_query(self._norm(hidden).to(self.semantic_query.weight.dtype)).float()
                semantic_mask = state.semantic_mask
                context = (
                    self._hierarchical_context(query, state, plan)
                    if self.planner is not None
                    else self._native_context(query, state)
                )
                if self.semantic_head_gate is not None:
                    gain = 2 * torch.sigmoid(
                        self.semantic_head_gate(self._norm(hidden).to(self.semantic_head_gate.weight.dtype)).float()
                    )
                    if self.semantic_head_gate_position == "pre_norm":
                        context = (
                            context.reshape(*hidden.shape[:2], self.semantic_num_heads, self.semantic_head_dim)
                            * gain[..., None]
                        ).flatten(-2)
            else:
                context = torch.matmul(log_attention.exp(), state.semantic_values.float())
            normalized_context = self._norm(context)
            gate = torch.sigmoid(
                self.semantic_gate(
                    torch.cat((query, normalized_context), dim=-1).to(self.semantic_gate.weight.dtype)
                ).float()
            )
            output_context = normalized_context
            if gain is not None and self.semantic_head_gate_position == "post_norm":
                # Normalize once, then gate without another normalization.
                # A shared reduction of all head gains now really attenuates
                # the correction; global gating reads the ungated evidence.
                output_context = (
                    normalized_context.reshape(*hidden.shape[:2], self.semantic_num_heads, self.semantic_head_dim)
                    * gain[..., None]
                ).flatten(-2)
            residual = self.semantic_output(output_context.to(self.semantic_output.weight.dtype)).float()
            delta = gate * residual
            generation_hidden = torch.where(
                semantic_mask.any(-1)[:, None, None],
                self._fuse(hidden, delta),
                hidden,
            )
        return generation_hidden, (log_attention, log_copy, log_generate)

    def output_logits(self, hidden: torch.Tensor, state: CopyState, lm_head, plan=None) -> torch.Tensor:
        generation_hidden, distribution = self.read(hidden, state, plan)
        return self._mix_logits(lm_head(generation_hidden), state, *distribution)

    def mix_logits(self, hidden: torch.Tensor, logits: torch.Tensor, state: CopyState) -> torch.Tensor:
        """Legacy copy-only API; semantic read must happen before the vocabulary head."""
        if self.semantic_read_enabled:
            raise ValueError("Use output_logits with semantic read so the LM sees the source context")
        return self._mix_logits(logits, state, *self.distribution(hidden, state))

    @staticmethod
    def _mix_logits(logits, state, log_attention, log_copy, log_generate):
        source_probability = logits.new_zeros(logits.shape, dtype=torch.float32).scatter_add(
            -1, state.token_ids[:, None, :].expand(-1, logits.shape[1], -1), log_attention.exp()
        )
        normalizer = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        log_source = source_probability.clamp_min(torch.finfo(torch.float32).tiny).log()
        log_source = log_source.masked_fill(source_probability.eq(0), torch.finfo(torch.float32).min)
        mixed = torch.logaddexp(logits.float() - normalizer + log_generate, log_source + log_copy) + normalizer
        return torch.where(state.mask.any(-1)[:, None, None], mixed, logits.float())

    def loss(self, hidden: torch.Tensor, labels: torch.Tensor, state: CopyState, lm_head, chunk_size: int, plan=None):
        def chunk_loss(states, targets, coverage=None, recent=None):
            chunk_plan = None if coverage is None else ReadPlan(coverage, recent)
            generation_hidden, (log_attention, log_copy, log_generate) = self.read(states, state, chunk_plan)
            matches = targets[..., None].eq(state.token_ids[:, None, :]) & state.mask[:, None, :]
            copy_target = torch.logsumexp(log_attention.masked_fill(~matches, torch.finfo(torch.float32).min), dim=-1)
            valid = targets.ne(-100)
            lm_logits = lm_head(generation_hidden[valid]).float()
            lm_target = -F.cross_entropy(lm_logits, targets[valid], reduction="none")
            return -torch.logaddexp(
                lm_target + log_generate[..., 0][valid], copy_target[valid] + log_copy[..., 0][valid]
            ).sum()

        losses = []
        stride = max(1, chunk_size // hidden.shape[0])
        for start in range(0, hidden.shape[1], stride):
            states, targets = hidden[:, start : start + stride], labels[:, start : start + stride]
            extra = (
                ()
                if plan is None
                else (plan.coverage[:, start : start + stride], plan.recent[:, start : start + stride])
            )
            losses.append(
                checkpoint(chunk_loss, states, targets, *extra, use_reentrant=False)
                if torch.is_grad_enabled()
                else chunk_loss(states, targets, *extra)
            )
        total = torch.stack(losses).sum() if losses else hidden.sum() * 0.0
        return total / labels.ne(-100).sum().clamp_min(1)
