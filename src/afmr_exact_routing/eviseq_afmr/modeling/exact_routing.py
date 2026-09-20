"""Query-dependent hierarchical routing over exact source tokens.

The router keeps the source sequence as the value anchor. It adds a
query-dependent retrieval path: a decoder query first selects a
deterministic non-overlapping source block and then selects a token inside
that block.  The resulting joint distribution is used to read the original
projected source values.  No source-only prior or hard selection is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExactRoutingState:
    """Precomputed token/block keys and the untouched token value anchor."""

    token_keys: torch.Tensor
    block_keys: torch.Tensor
    values: torch.Tensor
    token_mask: torch.Tensor
    block_mask: torch.Tensor
    block_size: int

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def source_length(self) -> int:
        return int(self.values.shape[1])

    def index_select(self, indices: torch.Tensor) -> "ExactRoutingState":
        return ExactRoutingState(
            self.token_keys.index_select(0, indices),
            self.block_keys.index_select(0, indices),
            self.values.index_select(0, indices),
            self.token_mask.index_select(0, indices),
            self.block_mask.index_select(0, indices),
            self.block_size,
        )

    def with_values(self, values: torch.Tensor) -> "ExactRoutingState":
        return ExactRoutingState(
            self.token_keys,
            self.block_keys,
            values,
            self.token_mask,
            self.block_mask,
            self.block_size,
        )


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax that returns an all-zero row when no item is valid."""

    mask = mask.bool()
    finite_min = torch.finfo(logits.dtype).min
    probabilities = torch.softmax(logits.masked_fill(~mask, finite_min), dim=dim)
    probabilities = probabilities * mask.to(probabilities.dtype)
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(1.0e-12)


class ExactTokenRouter(nn.Module):
    """Factorized block/token attention with query chunking.

    ``token_keys`` are projected from every source token, while ``values``
    remain the direct-projected encoder states.  Blocks are created by source
    position and never overlap, so block identity is deterministic and does
    not introduce a learned/static source bias.
    """

    def __init__(self, hidden_size: int, rank: int = 128, block_size: int = 128, query_chunk_size: int = 128):
        super().__init__()
        if hidden_size <= 0 or rank <= 0 or block_size <= 0 or query_chunk_size <= 0:
            raise ValueError("Exact routing dimensions must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.block_size = int(block_size)
        self.query_chunk_size = int(query_chunk_size)
        self.query_proj = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.key_proj = nn.Linear(self.hidden_size, self.rank, bias=False)
        # Attention divides the dot product by sqrt(rank); each projection
        # must preserve unit-scale inputs across the *hidden* input width.
        nn.init.normal_(self.query_proj.weight, std=self.hidden_size**-0.5)
        nn.init.normal_(self.key_proj.weight, std=self.hidden_size**-0.5)

    def build(self, memory: torch.Tensor, token_mask: torch.Tensor) -> ExactRoutingState:
        if memory.ndim != 3:
            raise ValueError("memory must be [batch, source_tokens, hidden]")
        if token_mask.shape != memory.shape[:2]:
            raise ValueError("token_mask must match memory positions")
        token_mask = token_mask.bool()
        token_keys = self.key_proj(memory)
        length = int(memory.shape[1])
        blocks = max(1, (length + self.block_size - 1) // self.block_size)
        padded = blocks * self.block_size
        if padded != length:
            token_keys = F.pad(token_keys, (0, 0, 0, padded - length))
            padded_mask = F.pad(token_mask, (0, padded - length), value=False)
        else:
            padded_mask = token_mask
        grouped_keys = token_keys.view(memory.shape[0], blocks, self.block_size, self.rank)
        grouped_mask = padded_mask.view(memory.shape[0], blocks, self.block_size)
        counts = grouped_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(grouped_keys.dtype)
        block_keys = (grouped_keys * grouped_mask[..., None].to(grouped_keys.dtype)).sum(dim=2) / counts
        block_mask = grouped_mask.any(dim=-1)
        return ExactRoutingState(
            token_keys,
            block_keys,
            memory,
            token_mask,
            block_mask,
            self.block_size,
        )

    def forward(self, query: torch.Tensor, state: ExactRoutingState) -> torch.Tensor:
        if query.ndim != 3:
            raise ValueError("query must be [batch, query_tokens, hidden]")
        if query.shape[0] != state.batch_size:
            raise ValueError("query and routing state batch sizes must match")
        query_keys = self.query_proj(query)
        scale = self.rank**-0.5
        block_keys = state.block_keys.to(dtype=query_keys.dtype)
        token_keys = state.token_keys.to(dtype=query_keys.dtype)
        values = state.values.to(dtype=query.dtype)
        blocks = block_keys.shape[1]
        padded = blocks * state.block_size
        grouped_keys = token_keys.view(query.shape[0], blocks, state.block_size, self.rank)
        grouped_mask = state.token_mask
        if padded != state.source_length:
            grouped_mask = F.pad(grouped_mask, (0, padded - state.source_length), value=False)
        grouped_mask = grouped_mask.view(query.shape[0], blocks, state.block_size)
        output = query.new_zeros(query.shape[0], query.shape[1], self.hidden_size)
        for start in range(0, query.shape[1], self.query_chunk_size):
            query_chunk = query_keys[:, start : start + self.query_chunk_size]
            block_logits = torch.einsum("bqr,bkr->bqk", query_chunk, block_keys) * scale
            # A short final/content-masked block must not receive the same
            # prior mass as a full block when all query/key scores are equal.
            valid_counts = grouped_mask.sum(dim=-1).clamp_min(1)
            block_logits = block_logits + valid_counts.log()[:, None, :]
            block_prob = _masked_softmax(block_logits, state.block_mask[:, None, :], dim=-1)
            token_logits = torch.einsum("bqr,bkpr->bqkp", query_chunk, grouped_keys) * scale
            token_prob = _masked_softmax(token_logits, grouped_mask[:, None, :, :], dim=-1)
            joint = block_prob[..., None] * token_prob
            weights = joint.reshape(query.shape[0], query_chunk.shape[1], padded)
            weights = weights[..., : state.source_length]
            output[:, start : start + query_chunk.shape[1]] = torch.einsum("bqs,bsd->bqd", weights, values)
        return output


class ExactRoutingBridge(nn.Module):
    """Direct-projection bridge plus an optional exact-token route bank."""

    bridge_mode = "afmr"

    def __init__(
        self,
        encoder_hidden: int,
        decoder_hidden: int,
        config: dict,
        route_builder: Optional[ExactTokenRouter],
    ):
        super().__init__()
        self.bridge_mode = str(config.get("bridge_mode", "afmr"))
        if self.bridge_mode not in {"afmr", "direct_projection"}:
            raise ValueError("architecture.bridge_mode must be afmr or direct_projection")
        self.encoder_hidden = int(encoder_hidden)
        self.decoder_hidden = int(decoder_hidden)
        self.controller_dim = int(config.get("controller_dim", 256))
        self.route_builder = route_builder if self.bridge_mode == "afmr" else None
        if self.encoder_hidden == self.decoder_hidden:
            self.direct_projection: nn.Module = nn.Identity()
        else:
            self.direct_projection = nn.Linear(self.encoder_hidden, self.decoder_hidden, bias=False)
            nn.init.orthogonal_(self.direct_projection.weight)

    def forward(
        self,
        encoder_state,
        prompt_embeddings: torch.Tensor,
        prompt_mask: torch.Tensor,
        output_budget: torch.Tensor,
    ):
        del prompt_embeddings, output_budget
        final = encoder_state.final
        if final.ndim != 3:
            raise ValueError("encoder_state.final must be [batch, source_tokens, hidden]")
        if encoder_state.attention_mask.shape != final.shape[:2]:
            raise ValueError("encoder attention mask must match encoder states")
        if prompt_mask.ndim != 2 or prompt_mask.shape[0] != final.shape[0]:
            raise ValueError("prompt_mask must have the bridge batch dimension")
        memory_mask = encoder_state.attention_mask.bool()
        content = encoder_state.content_mask.bool() & memory_mask
        memory = self.direct_projection(final.float())
        memory = memory.masked_fill(~memory_mask.unsqueeze(-1), 0)
        source_bias = torch.zeros(final.shape[:2], device=final.device, dtype=torch.float32)
        controller = torch.zeros(final.shape[0], self.controller_dim, device=final.device, dtype=torch.float32)
        route = self.route_builder.build(memory, content) if self.route_builder is not None else None
        from .outputs import BridgeState

        return BridgeState(memory, memory_mask, content, source_bias, controller, exact_route=route)
