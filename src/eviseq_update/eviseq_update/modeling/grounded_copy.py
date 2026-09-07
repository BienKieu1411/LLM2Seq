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

    def index_select(self, indices: torch.Tensor) -> CopyState:
        return CopyState(
            *(value.index_select(0, indices) for value in (self.keys, self.token_ids, self.mask, self.bias)),
            None if self.semantic_values is None else self.semantic_values.index_select(0, indices),
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
        self.semantic_read_enabled = bool(semantic_read)
        self.semantic_value = self.semantic_output = self.semantic_gate = None
        if self.semantic_read_enabled:
            # Keep downstream initialization identical to the copy-only control.
            with torch.random.fork_rng(devices=[]):
                self.semantic_value = nn.Linear(hidden_size, semantic_rank, bias=False)
                self.semantic_output = nn.Linear(semantic_rank, hidden_size, bias=False)
                self.semantic_gate = nn.Linear(key_dim + semantic_rank, 1)
                nn.init.zeros_(self.semantic_output.weight)
                nn.init.zeros_(self.semantic_gate.weight)
                nn.init.constant_(self.semantic_gate.bias, math.log(semantic_gate_init / (1 - semantic_gate_init)))

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
        if self.semantic_read_enabled:
            semantic_values = overlap_pool(self.semantic_value(normalized_memory.to(self.semantic_value.weight.dtype)))
        return CopyState(keys, copy_token_ids, copy_token_mask & totals.gt(0), bias, semantic_values)

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
        """Use one attention distribution for both exact copy and LM conditioning."""
        query, log_attention, log_copy, log_generate = self._attention(hidden, state)
        generation_hidden = hidden
        if self.semantic_read_enabled:
            if state.semantic_values is None:
                raise ValueError("Semantic read is enabled but cached source values are missing")
            context = torch.matmul(log_attention.exp(), state.semantic_values.float())
            normalized_context = self._norm(context)
            gate = torch.sigmoid(
                self.semantic_gate(
                    torch.cat((query, normalized_context), dim=-1).to(self.semantic_gate.weight.dtype)
                ).float()
            )
            residual = self.semantic_output(normalized_context.to(self.semantic_output.weight.dtype)).float()
            generation_hidden = torch.where(
                state.mask.any(-1)[:, None, None],
                (hidden.float() + gate * residual).to(hidden.dtype),
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

    def loss(self, hidden: torch.Tensor, labels: torch.Tensor, state: CopyState, lm_head, chunk_size: int):
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
        return total / labels.ne(-100).sum().clamp_min(1)
