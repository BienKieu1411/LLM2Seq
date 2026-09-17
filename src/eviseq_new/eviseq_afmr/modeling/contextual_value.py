"""Source-only region context for a token-aligned cross-attention value residual."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ContextualValueBridge(nn.Module):
    def __init__(self, hidden_size: int, settings: dict, gradient_checkpointing: bool = False):
        super().__init__()
        dim, heads = int(settings["dim"]), int(settings["num_heads"])
        self.window_size = int(settings["window_size"])
        self.stride = int(settings["stride"])
        self.gradient_checkpointing = gradient_checkpointing
        self.input_norm = nn.RMSNorm(hidden_size, eps=1.0e-6)
        self.down = nn.Linear(hidden_size, dim, bias=False)
        self.region_norm = nn.RMSNorm(dim, eps=1.0e-6)
        self.region_attention = nn.MultiheadAttention(dim, heads, dropout=0.0, bias=False, batch_first=True)
        self.region_ffn = nn.Sequential(
            nn.RMSNorm(dim, eps=1.0e-6),
            nn.Linear(dim, 4 * dim, bias=False),
            nn.SiLU(),
            nn.Linear(4 * dim, dim, bias=False),
        )
        self.query_norm = nn.RMSNorm(dim, eps=1.0e-6)
        self.context_norm = nn.RMSNorm(dim, eps=1.0e-6)
        self.token_read = nn.MultiheadAttention(dim, heads, dropout=0.0, bias=False, batch_first=True)
        self.output = nn.Linear(dim, hidden_size, bias=False)
        nn.init.zeros_(self.output.weight)
        self.register_buffer(
            "position_frequencies",
            torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2).float() / dim),
            persistent=False,
        )

    def _pool(self, tokens: torch.Tensor, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, dim = tokens.shape
        count = content.sum(-1, keepdim=True)
        index = (content.long().cumsum(-1) - 1).clamp_min(0)
        compact = tokens.new_zeros(batch, length, dim, dtype=torch.float32).scatter_add(
            1, index[..., None].expand_as(tokens), tokens.float().masked_fill(~content[..., None], 0.0)
        )
        cumulative = F.pad(compact.cumsum(1), (0, 0, 1, 0))
        # Append a shifted final full window; mask it when it duplicates a regular one.
        regular = torch.arange(0, max(1, length - self.window_size + 1), self.stride, device=tokens.device)
        regular = regular[None].expand(batch, -1)
        last = (count - self.window_size).clamp_min(0)
        starts = torch.cat((regular, last), dim=1)
        valid = torch.cat((regular <= last, last.remainder(self.stride) != 0), dim=1)
        ends = torch.minimum(starts + self.window_size, count)
        starts = torch.minimum(starts, count)
        sizes = ends - starts
        valid = valid & (sizes > 0)
        pooled = cumulative.gather(1, ends[..., None].expand(-1, -1, dim)) - cumulative.gather(
            1, starts[..., None].expand(-1, -1, dim)
        )
        pooled = pooled / sizes.clamp_min(1)[..., None]
        positions = (starts.float() + (sizes.float() - 1).clamp_min(0) / 2) / self.stride
        return pooled.masked_fill(~valid[..., None], 0.0), valid, positions

    def _forward(self, anchor: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        tokens = self.down(self.input_norm(anchor.float()))
        pooled, valid, positions = self._pool(tokens, content)
        angles = positions[..., None] * self.position_frequencies.float()
        positional = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        regions = (pooled + positional).masked_fill(~valid[..., None], 0.0)
        # A dummy key keeps SDPA finite for rows with no source content. Their
        # residual is masked to zero below; no prefix/padding enters the pool.
        safe_valid = valid.clone()
        safe_valid[:, 0] = safe_valid[:, 0] | ~valid.any(-1)
        normalized = self.region_norm(regions)
        contextual, _ = self.region_attention(
            normalized, normalized, normalized, key_padding_mask=~safe_valid, need_weights=False
        )
        regions = regions + contextual
        regions = (regions + self.region_ffn(regions)).masked_fill(~valid[..., None], 0.0)
        region_kv = self.context_norm(regions)
        read, _ = self.token_read(
            self.query_norm(tokens.float()), region_kv, region_kv, key_padding_mask=~safe_valid, need_weights=False
        )
        return self.output(read).masked_fill(~content[..., None], 0.0)

    def forward(self, anchor: torch.Tensor, content_mask: torch.Tensor) -> torch.Tensor:
        content = content_mask.bool()
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(self._forward, anchor, content, use_reentrant=False)
        return self._forward(anchor, content)
