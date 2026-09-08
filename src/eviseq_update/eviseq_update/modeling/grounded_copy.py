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
    semantic_values: torch.Tensor | None = None
    semantic_keys: torch.Tensor | None = None
    semantic_mask: torch.Tensor | None = None
    semantic_bias: torch.Tensor | None = None

    def index_select(self, indices: torch.Tensor) -> CopyState:
        return CopyState(
            *(
                None if value is None else value.index_select(0, indices)
                for value in (
                    self.keys,
                    self.token_ids,
                    self.mask,
                    self.bias,
                    self.semantic_values,
                    self.semantic_keys,
                    self.semantic_mask,
                    self.semantic_bias,
                )
            ),
        )


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
        if semantic_attention not in {"shared_copy", "independent_source"}:
            raise ValueError("semantic_attention must be shared_copy or independent_source")
        if semantic_max_relative_rms is not None and not 0 < float(semantic_max_relative_rms) < math.inf:
            raise ValueError("semantic_max_relative_rms must be null or a finite positive number")
        self.semantic_read_enabled = bool(semantic_read)
        self.semantic_attention = semantic_attention
        self.semantic_max_relative_rms = None if semantic_max_relative_rms is None else float(semantic_max_relative_rms)
        self.semantic_value = self.semantic_output = self.semantic_gate = None
        self.semantic_key = self.semantic_query = None
        if self.semantic_read_enabled:
            # Keep downstream initialization identical to the copy-only control.
            with torch.random.fork_rng(devices=[]):
                self.semantic_value = nn.Linear(hidden_size, semantic_rank, bias=False)
                self.semantic_output = nn.Linear(semantic_rank, hidden_size, bias=False)
                query_dim = semantic_rank if semantic_attention == "independent_source" else key_dim
                self.semantic_gate = nn.Linear(query_dim + semantic_rank, 1)
                nn.init.zeros_(self.semantic_output.weight)
                nn.init.zeros_(self.semantic_gate.weight)
                nn.init.constant_(self.semantic_gate.bias, math.log(semantic_gate_init / (1 - semantic_gate_init)))
                if semantic_attention == "independent_source":
                    self.semantic_key = nn.Linear(hidden_size, semantic_rank, bias=False)
                    self.semantic_query = nn.Linear(hidden_size, semantic_rank, bias=False)

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
            if self.semantic_attention == "independent_source":
                # Read native encoder positions. Copy-token boundaries and lexical
                # embeddings must not decide which semantic values the LM reads.
                semantic_keys = self._norm(self.semantic_key(normalized_memory.to(self.semantic_key.weight.dtype)))
                semantic_mask = content_mask.bool()
                semantic_bias = source_bias.float()
            else:
                semantic_values = overlap_pool(semantic_values)
        return CopyState(
            keys,
            copy_token_ids,
            copy_token_mask & totals.gt(0),
            bias,
            semantic_values,
            semantic_keys,
            semantic_mask,
            semantic_bias,
        )

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

    def read(self, hidden: torch.Tensor, state: CopyState):
        """Condition the LM while preserving the exact-copy distribution.

        Missing new config fields retain the original shared, uncapped graph.
        """
        query, log_attention, log_copy, log_generate = self._attention(hidden, state)
        generation_hidden = hidden
        if self.semantic_read_enabled:
            if state.semantic_values is None:
                raise ValueError("Semantic read is enabled but cached source values are missing")
            semantic_mask = state.mask
            if self.semantic_attention == "independent_source":
                if state.semantic_keys is None or state.semantic_mask is None or state.semantic_bias is None:
                    raise ValueError("Independent semantic read requires cached native source keys, mask and bias")
                query = self.semantic_query(self._norm(hidden).to(self.semantic_query.weight.dtype)).float()
                scores = torch.matmul(query, state.semantic_keys.float().transpose(1, 2)) / math.sqrt(query.shape[-1])
                scores = scores + state.semantic_bias.float()[:, None, :]
                semantic_mask = state.semantic_mask
                floor = torch.finfo(torch.float32).min
                semantic_log_attention = F.log_softmax(scores.masked_fill(~semantic_mask[:, None, :], floor), dim=-1)
                semantic_log_attention = semantic_log_attention.masked_fill(~semantic_mask[:, None, :], floor)
            else:
                semantic_log_attention = log_attention
            context = torch.matmul(semantic_log_attention.exp(), state.semantic_values.float())
            normalized_context = self._norm(context)
            gate = torch.sigmoid(
                self.semantic_gate(
                    torch.cat((query, normalized_context), dim=-1).to(self.semantic_gate.weight.dtype)
                ).float()
            )
            residual = self.semantic_output(normalized_context.to(self.semantic_output.weight.dtype)).float()
            delta = gate * residual
            if self.semantic_max_relative_rms is not None:
                # Smoothly bound RMS(delta) <= rho * RMS(hidden), including
                # learned projection growth. A sigmoid gate alone cannot do so.
                # vector_norm has a defined zero gradient at the zero vector.
                cap = (
                    self.semantic_max_relative_rms
                    * torch.linalg.vector_norm(hidden.float(), dim=-1, keepdim=True)
                    / math.sqrt(hidden.shape[-1])
                )
                delta = delta * cap / torch.sqrt(cap.square() + delta.square().mean(-1, keepdim=True) + 1e-12)
            generation_hidden = torch.where(
                semantic_mask.any(-1)[:, None, None],
                (hidden.float() + delta).to(hidden.dtype),
                hidden,
            )
        return generation_hidden, (log_attention, log_copy, log_generate)

    def output_logits(self, hidden: torch.Tensor, state: CopyState, lm_head) -> torch.Tensor:
        generation_hidden, distribution = self.read(hidden, state)
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

    def loss_sum(self, hidden: torch.Tensor, labels: torch.Tensor, state: CopyState, lm_head, chunk_size: int):
        def chunk_loss(states, targets):
            generation_hidden, (log_attention, log_copy, log_generate) = self.read(states, state)
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
            losses.append(
                checkpoint(chunk_loss, states, targets, use_reentrant=False)
                if torch.is_grad_enabled()
                else chunk_loss(states, targets)
            )
        total = torch.stack(losses).sum() if losses else hidden.sum() * 0.0
        return total

    def loss(self, hidden: torch.Tensor, labels: torch.Tensor, state: CopyState, lm_head, chunk_size: int):
        return self.loss_sum(hidden, labels, state, lm_head, chunk_size) / labels.ne(-100).sum().clamp_min(1)

    @staticmethod
    def _set_logsumexp(values: torch.Tensor, owner: torch.Tensor, size: int) -> torch.Tensor:
        """Stable logsumexp over sparse candidate sets grouped by target owner."""

        maximum = torch.full((size,), float("-inf"), device=values.device, dtype=torch.float32)
        maximum.scatter_reduce_(0, owner, values.float(), reduce="amax", include_self=True)
        total = torch.zeros((size,), device=values.device, dtype=torch.float32)
        total.scatter_add_(0, owner, torch.exp(values.float() - maximum.index_select(0, owner)))
        return maximum + total.clamp_min(torch.finfo(torch.float32).tiny).log()

    def _contrastive_branch(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        unit_batch_index: torch.Tensor,
        unit_confidence: torch.Tensor,
        unit_valid: torch.Tensor,
        target_unit: torch.Tensor,
        target_weight: torch.Tensor,
        owner: torch.Tensor,
        source_position: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        """Set-based multi-positive InfoNCE on the actual read logits."""

        zero = query.sum() * 0.0
        target_count = query.shape[0]
        if target_count == 0:
            return zero
        if owner.numel() == 0:
            if unit_valid.any():
                raise ValueError("Active evidence unit has no candidates for an enabled contrastive branch")
            return zero
        if owner.ndim != source_position.ndim or owner.shape != source_position.shape or owner.shape != positive.shape:
            raise ValueError("Evidence candidate tensors must have matching one-dimensional shapes")
        if owner.numel() and (owner.min() < 0 or owner.max() >= target_count):
            raise ValueError("Evidence candidate owner lies outside target rows")
        target_active = unit_valid.index_select(0, target_unit).bool()
        candidate_active = target_active.index_select(0, owner)
        owner = owner[candidate_active]
        source_position = source_position[candidate_active]
        positive = positive[candidate_active]
        if owner.numel() == 0:
            return zero
        source_batch = unit_batch_index.index_select(0, target_unit).index_select(0, owner)
        if source_position.min() < 0 or source_position.max() >= keys.shape[1]:
            raise ValueError("Evidence source position lies outside the active read keys")
        if not mask[source_batch, source_position].all():
            raise ValueError("Evidence candidate points to a masked source position")
        active_owner = torch.nonzero(target_active, as_tuple=False).flatten()
        positives = torch.bincount(owner[positive], minlength=target_count)
        negatives = torch.bincount(owner[~positive], minlength=target_count)
        if (positives.index_select(0, active_owner) == 0).any() or (negatives.index_select(0, active_owner) == 0).any():
            raise ValueError("Every active evidence timestep needs both positive and negative candidates")
        score = (query.index_select(0, owner).float() * keys[source_batch, source_position].float()).sum(
            -1
        ) / math.sqrt(query.shape[-1]) + bias[source_batch, source_position].float()
        all_lse = self._set_logsumexp(score, owner, target_count)
        positive_lse = self._set_logsumexp(score[positive], owner[positive], target_count)
        loss = all_lse.index_select(0, active_owner) - positive_lse.index_select(0, active_owner)
        weight = (
            unit_confidence.index_select(0, target_unit.index_select(0, active_owner)).float()
            * target_weight.index_select(0, active_owner).float()
        )
        return (loss * weight).sum()

    def evidence_loss(
        self, hidden: torch.Tensor, state: CopyState, evidence: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return copy and semantic contrastive sums without global normalization."""

        required = (
            "evidence_unit_batch_index",
            "evidence_unit_confidence",
            "evidence_unit_valid",
            "evidence_target_unit",
            "evidence_target_hidden_pos",
            "evidence_target_weight",
        )
        missing = [key for key in required if key not in evidence]
        if missing:
            raise ValueError(f"Evidence batch is missing {missing}")
        unit_batch = evidence["evidence_unit_batch_index"]
        unit_confidence = evidence["evidence_unit_confidence"]
        unit_valid = evidence["evidence_unit_valid"]
        target_unit = evidence["evidence_target_unit"]
        target_hidden = evidence["evidence_target_hidden_pos"]
        target_weight = evidence["evidence_target_weight"]
        if unit_batch.ndim != 1 or target_unit.ndim != 1 or target_hidden.ndim != 1 or target_weight.ndim != 1:
            raise ValueError("Evidence unit and target tensors must be one-dimensional")
        if not (unit_batch.shape == unit_confidence.shape == unit_valid.shape):
            raise ValueError("Evidence unit tensors must have matching shapes")
        if not (target_unit.shape == target_hidden.shape == target_weight.shape):
            raise ValueError("Evidence target tensors must have matching shapes")
        zero = hidden.sum() * 0.0
        if target_unit.numel() == 0:
            return zero, zero
        if target_unit.min() < 0 or target_unit.max() >= unit_batch.numel():
            raise ValueError("Evidence target refers to an unknown unit")
        if target_hidden.min() < 0 or target_hidden.max() >= hidden.shape[1]:
            raise ValueError("Evidence target hidden position lies outside decoder states")
        batch = unit_batch.index_select(0, target_unit)
        query_hidden = hidden[batch, target_hidden]
        copy_query = self.query(self._norm(query_hidden).to(self.query.weight.dtype)).float()
        copy_owner = evidence.get("evidence_copy_owner")
        copy_sum = zero
        if copy_owner is not None and copy_owner.numel():
            copy_sum = self._contrastive_branch(
                copy_query,
                state.keys,
                state.bias,
                state.mask,
                unit_batch,
                unit_confidence,
                unit_valid,
                target_unit,
                target_weight,
                copy_owner,
                evidence["evidence_copy_source_position"],
                evidence["evidence_copy_positive"],
            )
        semantic_sum = zero
        semantic_owner = evidence.get("evidence_semantic_owner")
        if semantic_owner is not None and semantic_owner.numel():
            if not self.semantic_read_enabled or self.semantic_attention != "independent_source":
                raise ValueError("Evidence semantic contrastive training requires independent semantic read")
            if state.semantic_keys is None or state.semantic_mask is None or state.semantic_bias is None:
                raise ValueError("Semantic evidence requires native source keys, mask and bias")
            semantic_query = self.semantic_query(self._norm(query_hidden).to(self.semantic_query.weight.dtype)).float()
            semantic_sum = self._contrastive_branch(
                semantic_query,
                state.semantic_keys,
                state.semantic_bias,
                state.semantic_mask,
                unit_batch,
                unit_confidence,
                unit_valid,
                target_unit,
                target_weight,
                semantic_owner,
                evidence["evidence_semantic_source_position"],
                evidence["evidence_semantic_positive"],
            )
        return copy_sum, semantic_sum
