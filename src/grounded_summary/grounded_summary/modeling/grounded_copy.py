from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_NEG_INF = torch.finfo(torch.float32).min


@dataclass
class CopyState:
    keys: torch.Tensor
    token_ids: torch.Tensor
    mask: torch.Tensor
    bias: torch.Tensor

    def __post_init__(self) -> None:
        if self.keys.ndim != 3:
            raise ValueError("CopyState.keys must be [batch, candidates, key_dim]")
        if self.keys.shape[1] <= 0:
            raise ValueError("CopyState requires at least one candidate slot")
        expected = self.keys.shape[:2]
        if self.token_ids.shape != expected or self.mask.shape != expected or self.bias.shape != expected:
            raise ValueError("CopyState token_ids, mask and bias must match keys [batch, candidates]")
        if self.token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("CopyState.token_ids must be an integer tensor")
        devices = {self.keys.device, self.token_ids.device, self.mask.device, self.bias.device}
        if len(devices) != 1:
            raise ValueError("CopyState tensors must share one device")

    def index_select(self, indices: torch.Tensor) -> CopyState:
        return CopyState(
            *(value.index_select(0, indices) for value in (self.keys, self.token_ids, self.mask, self.bias))
        )


class GroundedCopyHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        key_dim: int,
        gate_init: float,
        alpha_max: float = 1.0,
        generate_reserve: float = 0.0,
    ):
        super().__init__()
        if hidden_size <= 0 or key_dim <= 0:
            raise ValueError("Grounded copy requires hidden_size and key_dim > 0")
        if not 0 < float(alpha_max) <= 1:
            raise ValueError("Grounded copy alpha_max must lie in (0, 1]")
        if not 0 <= float(generate_reserve) < 1:
            raise ValueError("Grounded copy generate_reserve must lie in [0, 1)")
        effective_max = min(float(alpha_max), 1.0 - float(generate_reserve))
        if not 0 < float(gate_init) < effective_max:
            raise ValueError("Grounded copy gate_init must lie below the effective alpha_max")
        self.hidden_size = int(hidden_size)
        self.key_dim = int(key_dim)
        self.alpha_max = effective_max
        self.generate_reserve = float(generate_reserve)
        self.query = nn.Linear(hidden_size, key_dim, bias=False)
        self.context_key = nn.Linear(hidden_size, key_dim, bias=False)
        self.lexical_key = nn.Linear(hidden_size, key_dim, bias=False)
        self.gate = nn.Linear(2 * key_dim, 1)
        nn.init.zeros_(self.gate.weight)
        init_ratio = float(gate_init) / self.alpha_max
        nn.init.constant_(self.gate.bias, math.log(init_ratio / (1.0 - init_ratio)))

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
        self._validate_prepare_inputs(
            memory,
            source_bias,
            content_mask,
            embedding,
            copy_token_ids,
            copy_token_mask,
            copy_encoder_indices,
            copy_token_indices,
            copy_alignment_weights,
        )
        batch, width = copy_token_ids.shape
        copy_token_ids = copy_token_ids.long()
        copy_encoder_indices = copy_encoder_indices.long()
        copy_token_indices = copy_token_indices.long()
        copy_token_mask = copy_token_mask.bool()
        destination_valid = copy_token_mask.gather(1, copy_token_indices)
        valid = content_mask.bool().gather(1, copy_encoder_indices)
        weights = copy_alignment_weights.float() * valid.float() * destination_valid.float()

        context = self.context_key(self._norm(memory).to(self.context_key.weight.dtype)).float()
        rank = context.shape[-1]
        destination = copy_token_indices[..., None].expand(-1, -1, rank)
        pooled = memory.new_zeros(batch, width, rank, dtype=torch.float32).scatter_add(
            1,
            destination,
            context.gather(1, copy_encoder_indices[..., None].expand(-1, -1, rank)) * weights[..., None],
        )
        totals = weights.new_zeros(batch, width).scatter_add(1, copy_token_indices, weights)
        active = copy_token_mask & totals.gt(0)
        pooled = pooled / totals.clamp_min(1e-8)[..., None]
        bias = source_bias.new_zeros(batch, width).float().scatter_add(
            1, copy_token_indices, source_bias.float().gather(1, copy_encoder_indices) * weights
        ) / totals.clamp_min(1e-8)
        lexical = self._lexical_features(copy_token_ids, active, embedding, rank)
        keys = self._norm(pooled + lexical)
        keys = keys.masked_fill(~active[..., None], 0.0)
        bias = bias.masked_fill(~active, 0.0)
        return CopyState(keys, copy_token_ids.long(), active, bias)

    def _validate_prepare_inputs(
        self,
        memory: torch.Tensor,
        source_bias: torch.Tensor,
        content_mask: torch.Tensor,
        embedding: nn.Module,
        copy_token_ids: torch.Tensor,
        copy_token_mask: torch.Tensor,
        copy_encoder_indices: torch.Tensor,
        copy_token_indices: torch.Tensor,
        copy_alignment_weights: torch.Tensor,
    ) -> None:
        if memory is None:
            raise ValueError("Grounded copy requires bridge.value_memory (H0)")
        if memory.ndim != 3 or memory.shape[-1] != self.hidden_size:
            raise ValueError("memory must be [batch, source_tokens, hidden_size]")
        if source_bias.shape != memory.shape[:2] or content_mask.shape != memory.shape[:2]:
            raise ValueError("source_bias and content_mask must match memory [batch, source_tokens]")
        if copy_token_ids.ndim != 2 or copy_token_mask.shape != copy_token_ids.shape:
            raise ValueError("copy token ids/mask must be [batch, candidates]")
        if copy_token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("copy_token_ids must be an integer tensor")
        batch, width = copy_token_ids.shape
        if copy_token_ids.shape[0] != memory.shape[0]:
            raise ValueError("copy tensors must have the same batch size as memory")
        alignment = (copy_encoder_indices, copy_token_indices, copy_alignment_weights)
        if any(t.ndim != 2 or t.shape[0] != batch for t in alignment):
            raise ValueError("copy alignment tensors must be [batch, edges]")
        edge_shape = copy_encoder_indices.shape
        if any(t.shape != edge_shape for t in alignment):
            raise ValueError("copy alignment tensors must share one [batch, edges] shape")
        if copy_encoder_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("copy_encoder_indices must be an integer tensor")
        if copy_token_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("copy_token_indices must be an integer tensor")
        devices = {
            memory.device,
            source_bias.device,
            content_mask.device,
            copy_token_ids.device,
            copy_token_mask.device,
            copy_encoder_indices.device,
            copy_token_indices.device,
            copy_alignment_weights.device,
        }
        if len(devices) != 1:
            raise ValueError("memory, masks and copy tensors must be on the same device")
        # Collation validates value ranges on CPU. Avoid host synchronizations on
        # every CUDA batch; structural checks below are asynchronous-safe.
        check_values = memory.device.type == "cpu"
        if check_values and copy_encoder_indices.numel():
            if int(copy_encoder_indices.min()) < 0 or int(copy_encoder_indices.max()) >= memory.shape[1]:
                raise ValueError("copy encoder index is outside source memory")
        if check_values and copy_token_indices.numel():
            if int(copy_token_indices.min()) < 0 or int(copy_token_indices.max()) >= width:
                raise ValueError("copy token index is outside candidate width")
        if check_values and copy_token_ids.numel() and int(copy_token_ids.min()) < 0:
            raise ValueError("copy token IDs must be non-negative")
        if (
            check_values
            and copy_alignment_weights.numel()
            and (
                not torch.isfinite(copy_alignment_weights.float()).all()
                or bool(copy_alignment_weights.float().lt(0).any())
            )
        ):
            raise ValueError("copy alignment weights must be finite and non-negative")
        if check_values and not torch.isfinite(source_bias.float()).all():
            raise ValueError("source_bias must be finite")
        num_embeddings = getattr(embedding, "num_embeddings", None)
        if (
            check_values
            and num_embeddings is not None
            and copy_token_ids.numel()
            and int(copy_token_ids.max()) >= int(num_embeddings)
        ):
            raise ValueError("copy token ID is outside the decoder vocabulary")

    def _lexical_features(
        self,
        token_ids: torch.Tensor,
        active: torch.Tensor,
        embedding: nn.Module,
        rank: int,
    ) -> torch.Tensor:
        lexical = token_ids.new_zeros((*token_ids.shape, rank), dtype=torch.float32)
        active_positions = active.reshape(-1).nonzero(as_tuple=False).flatten()
        if active_positions.numel() == 0:
            return lexical
        active_ids = token_ids.reshape(-1).index_select(0, active_positions)
        unique_ids, inverse = torch.unique(active_ids, return_inverse=True)
        lexical_bank = self.lexical_key(self._norm(embedding(unique_ids)).to(self.lexical_key.weight.dtype)).float()
        lexical_flat = lexical.reshape(-1, rank)
        return lexical_flat.index_copy(0, active_positions, lexical_bank.index_select(0, inverse)).reshape_as(lexical)

    def distribution(self, hidden: torch.Tensor, state: CopyState):
        if (
            hidden.ndim != 3
            or hidden.shape[-1] != self.hidden_size
            or hidden.shape[0] != state.keys.shape[0]
            or state.keys.shape[-1] != self.key_dim
        ):
            raise ValueError("hidden and CopyState must match [batch, time, hidden_size] and key_dim")
        mask = state.mask.bool()
        query = self.query(self._norm(hidden).to(self.query.weight.dtype)).float()
        scores = torch.matmul(query, state.keys.float().transpose(1, 2)) / math.sqrt(query.shape[-1])
        scores = scores + state.bias.float()[:, None, :]
        valid = mask[:, None, :]
        has_source = mask.any(-1)[:, None, None]
        safe_scores = scores.masked_fill(~valid, _NEG_INF)
        safe_scores = torch.where(has_source, safe_scores, torch.zeros_like(safe_scores))
        log_attention = F.log_softmax(safe_scores, dim=-1).masked_fill(~valid, _NEG_INF)
        context = torch.matmul(log_attention.exp(), state.keys.float())
        gate = self.gate(torch.cat((query, context), dim=-1).to(self.gate.weight.dtype)).float()
        alpha = self.alpha_max * torch.sigmoid(gate)
        alpha = alpha.clamp_max(1.0 - torch.finfo(alpha.dtype).eps)
        log_copy = torch.where(has_source, alpha.clamp_min(torch.finfo(alpha.dtype).tiny).log(), _NEG_INF)
        log_generate = torch.where(has_source, torch.log1p(-alpha), 0.0)
        return log_attention, log_copy, log_generate

    def mix_logits(self, hidden: torch.Tensor, logits: torch.Tensor, state: CopyState) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[:2] != hidden.shape[:2]:
            raise ValueError("logits must be [batch, time, vocab] and match hidden")
        log_attention, log_copy, log_generate = self.distribution(hidden, state)
        source_probability = logits.new_zeros(logits.shape, dtype=torch.float32).scatter_add(
            -1, state.token_ids[:, None, :].expand(-1, hidden.shape[1], -1), log_attention.exp()
        )
        normalizer = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        log_source = torch.where(
            source_probability.gt(0), source_probability.clamp_min(torch.finfo(torch.float32).tiny).log(), _NEG_INF
        )
        mixed = torch.logaddexp(logits.float() - normalizer + log_generate, log_source + log_copy) + normalizer
        return torch.where(state.mask.bool().any(-1)[:, None, None], mixed, logits.float())

    def loss(self, hidden: torch.Tensor, labels: torch.Tensor, state: CopyState, lm_head, chunk_size: int):
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size or hidden.shape[0] != state.keys.shape[0]:
            raise ValueError("hidden must be [batch, time, hidden_size] and match CopyState batch")
        if labels.shape != hidden.shape[:2]:
            raise ValueError("labels must match hidden batch and time dimensions")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        def chunk_loss(states, targets):
            valid = targets.ne(-100)
            if not bool(valid.any()):
                return states.sum() * 0.0
            log_attention, log_copy, log_generate = self.distribution(states, state)
            matches = targets[..., None].eq(state.token_ids[:, None, :]) & state.mask.bool()[:, None, :]
            copy_target = torch.logsumexp(log_attention.masked_fill(~matches, _NEG_INF), dim=-1)
            lm_logits = lm_head(states[valid]).float()
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
