"""Lightweight cascaded future-token predictor for verified decoding.

The module deliberately owns neither the decoder embedding nor the language
model head.  Phase 3 reuses those frozen pretrained weights, so its checkpoint
contains only the small draft blocks instead of another vocabulary matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureTokenBlock(nn.Module):
    """Fuse one proposed token with a decoder state using a position-wise FFN.

    A self-attention layer is wasteful here: verified decoding calls each depth
    with one state, for which attention degenerates to a linear projection.
    The frozen main decoder has already contextualized the state.
    """

    def __init__(self, hidden_size: int, ffn_size: int, dropout: float) -> None:
        super().__init__()
        self.state_norm = nn.RMSNorm(hidden_size)
        self.token_norm = nn.RMSNorm(hidden_size)
        self.fuse = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.fused_norm = nn.RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.residual_gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, state: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        original_dtype = state.dtype
        parameter_dtype = self.fuse.weight.dtype
        state = state.to(parameter_dtype)
        token_embedding = token_embedding.to(parameter_dtype)
        fused = self.fuse(
            torch.cat([self.state_norm(state), self.token_norm(token_embedding)], dim=-1)
        )
        normalized = self.fused_norm(fused)
        update = self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        state = fused + torch.tanh(self.residual_gate) * self.dropout(update)
        return state.to(original_dtype)


class TiedLowRankDraftHead(nn.Module):
    """Cheap vocabulary projection tied to a fixed slice of the main LM head.

    Qwen3.5 has a large vocabulary, so running the full LM head once per draft
    depth can erase speculative-decoding gains.  This head learns only an
    H->R projection and reuses R evenly spaced columns from the frozen main LM
    head as its vocabulary codebook.  The large codebook is a derived,
    non-persistent cache and therefore is not duplicated in the checkpoint.
    """

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.rank = min(int(rank), int(hidden_size))
        if self.rank < 1:
            raise ValueError("mtp.draft_head_rank must be positive")
        indices = torch.linspace(0, hidden_size - 1, self.rank).round().long().unique()
        self.rank = int(indices.numel())
        self.register_buffer("column_indices", indices, persistent=True)
        self.norm = nn.RMSNorm(hidden_size)
        self.down_proj = nn.Linear(hidden_size, self.rank, bias=False)
        self._codebook: Optional[torch.Tensor] = None
        with torch.no_grad():
            self.down_proj.weight.zero_()
            self.down_proj.weight[
                torch.arange(self.rank), self.column_indices
            ] = 1.0

    @torch.no_grad()
    def prepare(self, lm_head: nn.Module) -> None:
        weight = lm_head.weight.detach()
        indices = self.column_indices.to(weight.device)
        self._codebook = weight.index_select(1, indices).contiguous()

    def clear(self) -> None:
        self._codebook = None

    def forward(self, hidden_states: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
        weight = lm_head.weight
        if (
            self._codebook is None
            or self._codebook.device != weight.device
            or self._codebook.dtype != weight.dtype
            or self._codebook.shape[0] != weight.shape[0]
        ):
            self.prepare(lm_head)
        hidden_states = hidden_states.to(self.down_proj.weight.dtype)
        projected = self.down_proj(self.norm(hidden_states))
        projected = projected.to(self._codebook.dtype)
        return F.linear(projected, self._codebook)


class CascadedFuturePredictor(nn.Module):
    """Predict consecutive future tokens from one frozen decoder state."""

    def __init__(self, hidden_size: int, config: Dict[str, object]) -> None:
        super().__init__()
        self.num_draft_tokens = int(config.get("num_draft_tokens", 4))
        if self.num_draft_tokens < 1:
            raise ValueError("mtp.num_draft_tokens must be positive")
        ffn_size = int(config.get("ffn_size", 2 * hidden_size))
        dropout = float(config.get("dropout", 0.05))
        draft_head_rank = int(config.get("draft_head_rank", 128))
        self.blocks = nn.ModuleList(
            FutureTokenBlock(hidden_size, ffn_size, dropout)
            for _ in range(self.num_draft_tokens)
        )
        self.draft_head = TiedLowRankDraftHead(hidden_size, draft_head_rank)
        self.apply(self._initialize)
        # TiedLowRankDraftHead has a meaningful identity-slice initializer.
        with torch.no_grad():
            self.draft_head.down_proj.weight.zero_()
            self.draft_head.down_proj.weight[
                torch.arange(self.draft_head.rank), self.draft_head.column_indices
            ] = 1.0

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def teacher_hidden_states(
        self,
        decoder_states: torch.Tensor,
        labels: torch.Tensor,
        embed_tokens: nn.Module,
        pad_token_id: int,
    ) -> List[torch.Tensor]:
        """Return depth states with ground-truth previous-token conditioning."""

        state = decoder_states.detach()
        outputs: List[torch.Tensor] = []
        sequence_length = labels.shape[1]
        for depth, block in enumerate(self.blocks):
            teacher_ids = torch.full_like(labels, int(pad_token_id))
            valid_width = sequence_length - depth
            if valid_width > 0:
                shifted = labels[:, depth : depth + valid_width]
                teacher_ids[:, :valid_width] = shifted.masked_fill(shifted.lt(0), int(pad_token_id))
            with torch.no_grad():
                token_embedding = embed_tokens(teacher_ids)
            state = block(state, token_embedding)
            outputs.append(state)
        return outputs

    @torch.no_grad()
    def draft(
        self,
        decoder_state: torch.Tensor,
        main_token: torch.Tensor,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        constrain: Callable[[torch.Tensor, Sequence[int]], torch.Tensor],
        prefix_tokens: Sequence[int],
        maximum: int | None = None,
    ) -> List[torch.Tensor]:
        """Draft tokens sequentially, applying the same greedy constraints."""

        limit = self.num_draft_tokens if maximum is None else min(self.num_draft_tokens, int(maximum))
        state = decoder_state
        previous = main_token
        drafted: List[torch.Tensor] = []
        speculative_prefix = list(prefix_tokens) + [int(main_token.item())]
        for block in self.blocks[:limit]:
            state = block(state, embed_tokens(previous))
            logits = self.draft_head(state[:, -1, :], lm_head)
            logits = constrain(logits, speculative_prefix)
            previous = logits.argmax(dim=-1, keepdim=True)
            drafted.append(previous)
            speculative_prefix.append(int(previous.item()))
        return drafted


def select_base_positions(
    labels: torch.Tensor,
    num_draft_tokens: int,
    maximum_per_sequence: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample base positions that have targets for every requested depth."""

    batch_indices: List[torch.Tensor] = []
    time_indices: List[torch.Tensor] = []
    required = num_draft_tokens + 1
    for batch_index in range(labels.shape[0]):
        valid = labels[batch_index].ne(-100)
        candidates = []
        for position in range(max(0, labels.shape[1] - required + 1)):
            if bool(valid[position : position + required].all()):
                candidates.append(position)
        if not candidates:
            continue
        indices = torch.tensor(candidates, device=labels.device, dtype=torch.long)
        if maximum_per_sequence > 0 and indices.numel() > maximum_per_sequence:
            order = torch.randperm(indices.numel(), device=labels.device)[:maximum_per_sequence]
            indices = indices[order]
        batch_indices.append(torch.full_like(indices, batch_index))
        time_indices.append(indices)
    if not batch_indices:
        empty = torch.empty(0, dtype=torch.long, device=labels.device)
        return empty, empty
    return torch.cat(batch_indices), torch.cat(time_indices)


def future_prediction_loss(
    predictor: CascadedFuturePredictor,
    decoder_states: torch.Tensor,
    labels: torch.Tensor,
    embed_tokens: nn.Module,
    lm_head: nn.Module,
    pad_token_id: int,
    config: Dict[str, object],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Mix main-model hard distillation with supervised future-token CE."""

    maximum = int(config.get("max_positions_per_sequence", 24))
    batch_index, time_index = select_base_positions(
        labels,
        predictor.num_draft_tokens,
        maximum,
    )
    if batch_index.numel() == 0:
        raise ValueError("Phase-3 batch has no sequence long enough for MTP targets")
    distill_weight = float(config.get("distill_weight", 0.8))
    supervised_weight = float(config.get("supervised_weight", 0.2))
    if distill_weight < 0 or supervised_weight < 0 or distill_weight + supervised_weight <= 0:
        raise ValueError("MTP loss weights must be non-negative with a positive sum")
    normalizer = distill_weight + supervised_weight
    distill_weight /= normalizer
    supervised_weight /= normalizer
    configured_depth_weights = list(config.get("depth_weights", [1.0, 0.8, 0.6, 0.4]))
    if len(configured_depth_weights) < predictor.num_draft_tokens:
        configured_depth_weights.extend(
            [configured_depth_weights[-1] if configured_depth_weights else 1.0]
            * (predictor.num_draft_tokens - len(configured_depth_weights))
        )

    losses = []
    metrics: Dict[str, torch.Tensor] = {}
    # Run the draft blocks only at sampled positions.  Applying them to the
    # complete padded target before selecting positions wastes most Phase-3
    # compute and memory.
    state = decoder_states[batch_index, time_index].detach()
    for depth, block in enumerate(predictor.blocks):
        conditioning_ids = labels[batch_index, time_index + depth]
        conditioning_ids = conditioning_ids.masked_fill(conditioning_ids.lt(0), int(pad_token_id))
        with torch.no_grad():
            conditioning_embeddings = embed_tokens(conditioning_ids)
        state = block(state, conditioning_embeddings)
        logits = predictor.draft_head(state, lm_head)
        future_time = time_index + depth + 1
        supervised_targets = labels[batch_index, future_time]
        with torch.no_grad():
            teacher_hidden = decoder_states[batch_index, future_time]
            teacher_targets = lm_head(teacher_hidden).argmax(dim=-1)
        hard_distill = F.cross_entropy(logits.float(), teacher_targets)
        supervised = F.cross_entropy(logits.float(), supervised_targets)
        depth_loss = distill_weight * hard_distill + supervised_weight * supervised
        losses.append(float(configured_depth_weights[depth]) * depth_loss)
        metrics[f"mtp_depth_{depth + 1}"] = depth_loss.detach()
        metrics[f"mtp_accept_proxy_{depth + 1}"] = logits.detach().argmax(dim=-1).eq(teacher_targets).float().mean()
    weight_sum = sum(float(value) for value in configured_depth_weights[: predictor.num_draft_tokens])
    total = torch.stack(losses).sum() / max(weight_sum, 1e-12)
    metrics["mtp_loss"] = total.detach()
    metrics["mtp_positions"] = torch.tensor(float(batch_index.numel()), device=labels.device)
    return total, metrics


def save_mtp_checkpoint(
    predictor: CascadedFuturePredictor,
    path: str | Path,
    config: Dict[str, Any],
    base_checkpoint: str,
    epoch: int,
    global_step: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dtype = str(config.get("checkpoint_dtype", "bfloat16")).lower()
    dtype_by_name = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if checkpoint_dtype not in dtype_by_name:
        raise ValueError(f"Unsupported mtp.checkpoint_dtype: {checkpoint_dtype}")
    storage_dtype = dtype_by_name[checkpoint_dtype]
    state = {}
    for name, tensor in predictor.state_dict().items():
        saved = tensor.detach().cpu()
        if saved.is_floating_point():
            saved = saved.to(storage_dtype)
        state[name] = saved
    torch.save(
        {
            "mtp_state_dict": state,
            "mtp_config": dict(config),
            "base_checkpoint": str(base_checkpoint),
            "epoch": int(epoch),
            "global_step": int(global_step),
        },
        path,
    )


def load_mtp_checkpoint(
    predictor: CascadedFuturePredictor,
    path: str | Path,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("mtp_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Not an EviBridge phase-3 checkpoint: {path}")
    predictor.load_state_dict(state, strict=True)
    return payload
