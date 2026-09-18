"""Decoder-query reads over a compact bank of overlapping source regions.

The bridge keeps the token-level source path intact.  This module is an
optional, per-decoder-layer residual read: keys are pooled from the semantic
memory ``M`` while values are pooled from the ``H0`` value anchor.  The region
bank is deliberately built outside the decoder loop so it can be cached during
autoregressive generation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_region_inputs(
    key_memory: torch.Tensor,
    value_memory: torch.Tensor,
    content_mask: torch.Tensor,
    window_size: int,
    stride: int,
) -> None:
    if key_memory.ndim != 3 or value_memory.ndim != 3:
        raise ValueError("key_memory and value_memory must have shape [batch, source_tokens, hidden]")
    if key_memory.shape[:2] != value_memory.shape[:2]:
        raise ValueError("key_memory and value_memory must have matching batch and source dimensions")
    if content_mask.shape != key_memory.shape[:2]:
        raise ValueError("content_mask must have shape [batch, source_tokens]")
    if isinstance(window_size, bool) or int(window_size) <= 0:
        raise ValueError("window_size must be a positive integer")
    if isinstance(stride, bool) or int(stride) <= 0:
        raise ValueError("stride must be a positive integer")


def _pool_one_memory(
    memory: torch.Tensor,
    content: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Pool compacted content-token windows without a source-token loop."""
    batch, length, hidden = memory.shape
    compact_index = (content.long().cumsum(-1) - 1).clamp_min(0)
    compact = memory.new_zeros(batch, length, hidden, dtype=torch.float32).scatter_add(
        1,
        compact_index[..., None].expand(-1, -1, hidden),
        memory.float().masked_fill(~content[..., None], 0.0),
    )
    cumulative = F.pad(compact.cumsum(1), (0, 0, 1, 0))
    pooled = cumulative.gather(1, ends[..., None].expand(-1, -1, hidden)) - cumulative.gather(
        1, starts[..., None].expand(-1, -1, hidden)
    )
    sizes = (ends - starts).to(pooled.dtype)
    pooled = pooled / sizes.clamp_min(1.0)[..., None]
    return pooled.masked_fill(~valid[..., None], 0.0).to(memory.dtype)


def pool_regions(
    key_memory: torch.Tensor,
    value_memory: torch.Tensor,
    content_mask: torch.Tensor,
    window_size: int = 128,
    stride: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a cacheable overlapping region bank.

    ``key_memory`` is the bridge memory ``M`` and ``value_memory`` is the
    source value anchor ``H0``.  Windows are measured over content tokens
    after prefix/padding positions are removed, so padding and decoder
    prompts cannot affect a region.  The returned tuple is
    ``(region_keys, region_values, region_mask)`` with shapes
    ``[B, R, H_key]``, ``[B, R, H_value]`` and ``[B, R]``.  ``R`` is shared by
    the batch and ``region_mask`` identifies rows whose last window is absent.

    An all-invalid row still receives one masked, zero-valued region.  This
    gives attention implementations a finite key/value row while guaranteeing
    a zero read.  The tuple can be retained and passed to
    :meth:`RegionQueryRead.prepare_cache` for autoregressive decoding.
    """
    window_size = int(window_size)
    stride = int(stride)
    _validate_region_inputs(key_memory, value_memory, content_mask, window_size, stride)
    content = content_mask.bool()
    batch, length, _ = key_memory.shape
    if length == 0:
        raise ValueError("source_tokens must be positive")

    counts = content.sum(dim=-1, keepdim=True)
    # ``regular`` is sized from the padded batch width; rows with fewer content
    # tokens are filtered by ``regular <= last`` below.
    regular = torch.arange(0, max(1, length - window_size + 1), stride, device=key_memory.device, dtype=torch.long)
    regular = regular[None, :].expand(batch, -1)
    last = (counts - window_size).clamp_min(0)
    starts = torch.cat((regular, last), dim=1)
    regular_valid = regular <= last
    # Append the final shifted window only when it is not already a regular
    # start.  ``sizes > 0`` below handles rows with no content.
    final_valid = last.remainder(stride).ne(0)
    valid = torch.cat((regular_valid, final_valid), dim=1)
    starts = torch.minimum(starts, counts)
    ends = torch.minimum(starts + window_size, counts)
    valid = valid & ends.gt(starts)

    region_keys = _pool_one_memory(key_memory, content, starts, ends, valid)
    region_values = _pool_one_memory(value_memory, content, starts, ends, valid)
    return region_keys, region_values, valid.bool()


class RegionQueryRead(nn.Module):
    """Low-rank, decoder-query-conditioned reads over pooled source regions.

    A separate instance is intended for each decoder layer.  Queries are
    projected from the layer's hidden states; keys come from pooled ``M`` and
    values from pooled ``H0``.  The output projection is zero initialized, so
    the module returns an exact zero tensor before its first optimizer update.
    The output lives in the copied cross-attention's projected Q space, before
    Q normalization.  Its perturbation is bounded independently for every
    query head relative to the unmodified projected Q.  The scalar gate is
    bounded by ``gate_max`` and initialized at ``gate_init``.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        gate_init: float = 0.05,
        gate_max: float = 0.20,
        num_heads: int = 1,
        num_query_heads: int = 1,
        query_head_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        rank = int(rank)
        num_heads = int(num_heads)
        num_query_heads = int(num_query_heads)
        query_head_dim = hidden_size if query_head_dim is None else int(query_head_dim)
        if hidden_size <= 0 or rank <= 0:
            raise ValueError("hidden_size and rank must be positive")
        if num_heads <= 0 or rank % num_heads:
            raise ValueError("rank must be divisible by a positive num_heads")
        if num_query_heads <= 0 or query_head_dim <= 0:
            raise ValueError("num_query_heads and query_head_dim must be positive")
        if not 0.0 < float(gate_init) < float(gate_max) <= 0.25:
            raise ValueError("gate_init must lie strictly between zero and gate_max <= 0.25")
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = rank // num_heads
        self.num_query_heads = num_query_heads
        self.query_head_dim = query_head_dim
        self.query_norm = nn.RMSNorm(hidden_size, eps=1.0e-6)
        self.key_norm = nn.RMSNorm(hidden_size, eps=1.0e-6)
        self.value_norm = nn.RMSNorm(hidden_size, eps=1.0e-6)
        self.q_proj = nn.Linear(hidden_size, rank, bias=False)
        self.k_proj = nn.Linear(hidden_size, rank, bias=False)
        self.v_proj = nn.Linear(hidden_size, rank, bias=False)
        self.out_proj = nn.Linear(rank, num_query_heads * query_head_dim, bias=False)
        # Zero output is the parity contract; q/k/v remain ordinary trainable
        # projections and receive gradients after the first output update.
        nn.init.zeros_(self.out_proj.weight)
        self.gate_max = float(gate_max)
        self.gate_raw = nn.Parameter(
            torch.tensor(math.log(float(gate_init) / (float(gate_max) - float(gate_init))), dtype=torch.float32)
        )
        self._cache: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def _gate(self) -> torch.Tensor:
        return self.gate_max * torch.sigmoid(self.gate_raw.float())

    def _project_regions(
        self,
        region_keys: torch.Tensor,
        region_values: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if region_keys.ndim != 3 or region_values.ndim != 3:
            raise ValueError("region keys and values must have shape [batch, regions, hidden]")
        if region_keys.shape != region_values.shape or region_keys.shape[-1] != self.hidden_size:
            raise ValueError("region keys and values must have shape [batch, regions, hidden_size]")
        if region_mask.shape != region_keys.shape[:2]:
            raise ValueError("region_mask must have shape [batch, regions]")
        mask = region_mask.bool()
        keys = region_keys.masked_fill(~mask[..., None], 0.0)
        values = region_values.masked_fill(~mask[..., None], 0.0)
        key = (
            self.k_proj(self.key_norm(keys))
            .view(keys.shape[0], keys.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(self.value_norm(values))
            .view(values.shape[0], values.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        return key, value, mask

    @torch.no_grad()
    def prepare_cache(
        self,
        region_keys: torch.Tensor,
        region_values: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> None:
        """Project and store one region bank for eval-time decoding."""
        key, value, mask = self._project_regions(region_keys, region_values, region_mask)
        self._cache = (key.contiguous(), value.contiguous(), mask.contiguous())

    def clear_cache(self) -> None:
        """Drop the eval-time projected region bank."""
        self._cache = None

    def select_cache(self, indices: torch.Tensor) -> None:
        """Compact cached rows after finished sequences leave a decode batch."""
        if self._cache is None:
            raise RuntimeError("RegionQueryRead cache is not prepared")
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("cache indices must be a one-dimensional integer tensor")
        self._cache = tuple(value.index_select(0, indices) for value in self._cache)  # type: ignore[assignment]

    def forward(
        self,
        query_states: torch.Tensor,
        q_base: torch.Tensor,
        region_keys: Optional[torch.Tensor] = None,
        region_values: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a per-head bounded ``[B,T,QH,HD]`` projected-Q residual."""
        if query_states.ndim != 3 or query_states.shape[-1] != self.hidden_size:
            raise ValueError("query_states must have shape [batch, query_tokens, hidden_size]")
        expected_q_shape = (
            query_states.shape[0],
            query_states.shape[1],
            self.num_query_heads,
            self.query_head_dim,
        )
        if q_base.shape != expected_q_shape:
            raise ValueError(f"q_base must have shape {expected_q_shape}")
        # Once prepared, projected K/V are reused for every decode step.  The
        # caller may still pass the original pooled tensors for API symmetry;
        # cache presence takes precedence until ``clear_cache`` is called.
        if self._cache is not None and not self.training:
            key, value, mask = self._cache
        else:
            if region_keys is None or region_values is None or region_mask is None:
                raise ValueError("region keys, values and mask are required when no eval cache is active")
            key, value, mask = self._project_regions(region_keys, region_values, region_mask)
        if key.shape[0] != query_states.shape[0]:
            raise ValueError("query and region banks must have the same batch size")
        query = (
            self.q_proj(self.query_norm(query_states))
            .view(query_states.shape[0], query_states.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        safe_mask = mask.clone()
        safe_mask[:, 0] |= ~safe_mask.any(dim=-1)
        # Invalid region tensors were zeroed in _project_regions.  The safe
        # dummy key therefore keeps SDPA finite while contributing zero value.
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=safe_mask[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(query_states.shape[0], query_states.shape[1], self.rank)
        raw_delta = self.out_proj(attended).view(expected_q_shape).float()
        # A bounded scalar gate alone does not bound the projection's learned
        # output.  Cap the *actual* per-token RMS perturbation relative to the
        # unmodified projected Q, separately for each token and attention
        # head.  Detach only the reference radius; q_base retains its normal
        # direct gradient through the copied cross-attention.
        query_rms = q_base.float().detach().square().mean(dim=-1, keepdim=True).sqrt()
        radius = self._gate() * query_rms
        raw_rms_sq = raw_delta.square().mean(dim=-1, keepdim=True)
        scale = radius / (radius.square() + raw_rms_sq + 1.0e-12).sqrt()
        delta = raw_delta * scale
        return delta.to(q_base.dtype)
