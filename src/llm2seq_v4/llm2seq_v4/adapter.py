"""Identity-preserving V4 bridge with prospective summary latent planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(states: torch.Tensor) -> torch.Tensor:
    first, second = states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class StableTokenLayerFusion(nn.Module):
    """Identity-preserving last-layer highway with a gated depth residual.

    The pretrained encoder's final hidden state is the *exact* base path.  A
    token-conditioned mixture of selected depths is allowed to influence that
    representation only through a small residual gate.  With the default zero
    gate this module is therefore an identity (apart from masking), which is an
    important invariant for low-data adaptation.
    """

    def __init__(
        self,
        hidden_size: int,
        layer_indices: Sequence[int],
        dropout: float,
        gate_init: float = 0.0,
    ):
        super().__init__()
        if not layer_indices:
            raise ValueError("Layer fusion requires at least one encoder layer")
        if not 0.0 <= gate_init < 1.0:
            raise ValueError("layer fusion gate initialization must be in [0, 1)")
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.norms = nn.ModuleList(nn.RMSNorm(hidden_size) for _ in self.layer_indices)
        initial = torch.zeros(len(self.layer_indices))
        self.global_logits = nn.Parameter(initial)
        scorer_hidden = max(64, hidden_size // 4)
        self.token_scorer = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, scorer_hidden, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(scorer_hidden, len(self.layer_indices), bias=True),
        )
        nn.init.zeros_(self.token_scorer[-1].weight)
        nn.init.zeros_(self.token_scorer[-1].bias)
        self.residual_gate = nn.Parameter(torch.tensor(math.atanh(gate_init), dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)
        self.last_mean_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        total = len(hidden_states)
        selected = []
        for norm, index in zip(self.norms, self.layer_indices):
            actual = index if index >= 0 else total + index
            if actual < 0 or actual >= total:
                raise IndexError(f"Encoder layer index {index} is outside {total} hidden states")
            selected.append(norm(hidden_states[actual]))
        stacked = torch.stack(selected, dim=2)  # [B, S, L, D]
        token_logits = self.token_scorer(hidden_states[-1])
        weights = F.softmax(token_logits.float() + self.global_logits.float(), dim=-1).to(stacked.dtype)
        depth_residual = torch.einsum("bsl,bsld->bsd", weights, stacked)
        base = hidden_states[-1]
        fused = base + torch.tanh(self.residual_gate).to(base.dtype) * self.dropout(depth_residual)
        fused = fused.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
        self.last_mean_weights = weights.detach().float().mean(dim=(0, 1))
        return fused


class IdentityResidualProjection(nn.Module):
    """Preserve encoder coordinates and learn only a gated task residual."""

    def __init__(self, input_size: int, output_size: int, ffn_size: int, dropout: float, gate_init: float):
        super().__init__()
        if not 0.0 <= gate_init < 1.0:
            raise ValueError("projection gate initialization must be in [0, 1)")
        self.same_width = input_size == output_size
        if self.same_width:
            # A fixed identity, not merely an identity initialization: full
            # fine-tuning cannot silently destroy the pretrained highway.
            self.base: nn.Module = nn.Identity()
        else:
            self.base = nn.Linear(input_size, output_size, bias=False)
            nn.init.xavier_uniform_(self.base.weight)
        self.delta_norm = nn.RMSNorm(output_size)
        self.gate_proj = nn.Linear(output_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(output_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, output_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.residual_gate = nn.Parameter(torch.tensor(math.atanh(gate_init), dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # Do not normalize the identity highway.  Normalization belongs only to
        # the learnable delta branch, so a same-width zero-gate projection is
        # bit-for-bit the pretrained representation.
        base = self.base(states)
        normalized = self.delta_norm(base)
        delta = self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        return base + torch.tanh(self.residual_gate).to(base.dtype) * self.dropout(delta)


class BidirectionalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, rope_theta: float):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.rope_theta = float(rope_theta)
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inverse = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, device=query.device, dtype=torch.float32) / self.head_dim)
        )
        frequencies = position_ids.float().unsqueeze(-1) * inverse
        embedding = torch.cat([frequencies, frequencies], dim=-1)
        cosine = embedding.cos().to(query.dtype).unsqueeze(1)
        sine = embedding.sin().to(query.dtype).unsqueeze(1)
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )

    def forward(
        self,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = states.shape
        query = self.q_proj(states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        query, key = self._rope(query, key, position_ids)
        valid = attention_mask.bool()[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.hidden_size)
        return self.o_proj(attended).masked_fill(~attention_mask.bool().unsqueeze(-1), 0)


class BidirectionalAdapterLayer(nn.Module):
    """Pre-norm full-attention block with small learned residual gates."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        dropout: float,
        rope_theta: float,
        gate_init: float,
    ):
        super().__init__()
        if not 0.0 <= gate_init < 1.0:
            raise ValueError("adapter gate initialization must be in [0, 1)")
        raw_gate = math.atanh(gate_init)
        self.attention_norm = nn.RMSNorm(hidden_size)
        self.attention = BidirectionalAttention(hidden_size, num_heads, dropout, rope_theta)
        self.attention_gate = nn.Parameter(torch.tensor(raw_gate, dtype=torch.float32))
        self.ffn_norm = nn.RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=False)
        self.ffn_gate = nn.Parameter(torch.tensor(raw_gate, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.attention(self.attention_norm(states), attention_mask, position_ids)
        states = states + torch.tanh(self.attention_gate).to(states.dtype) * self.dropout(attended)
        normalized = self.ffn_norm(states)
        update = self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        states = states + torch.tanh(self.ffn_gate).to(states.dtype) * self.dropout(update)
        return states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)


def _pool_units(
    token_states: torch.Tensor,
    unit_ids: torch.Tensor,
    unit_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, _, hidden = token_states.shape
    if unit_count <= 0:
        return (
            token_states.new_zeros((batch, 0, hidden)),
            torch.zeros((batch, 0), dtype=torch.bool, device=token_states.device),
        )
    sums = token_states.new_zeros((batch, unit_count + 1, hidden))
    counts = token_states.new_zeros((batch, unit_count + 1, 1))
    indices = unit_ids.clamp(min=0, max=unit_count)
    sums.scatter_add_(1, indices.unsqueeze(-1).expand(-1, -1, hidden), token_states)
    counts.scatter_add_(1, indices.unsqueeze(-1), torch.ones_like(token_states[..., :1]))
    pooled = sums[:, 1:] / counts[:, 1:].clamp_min(1.0)
    valid = counts[:, 1:, 0].gt(0)
    return pooled, valid


def _balanced_salience_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    supervised = valid & labels.ge(0)
    positive = supervised & labels.gt(0.5)
    negative = supervised & labels.le(0.5)
    terms = []
    if bool(positive.any()):
        terms.append(F.softplus(-logits[positive].float()).mean())
    if bool(negative.any()):
        terms.append(F.softplus(logits[negative].float()).mean())
    return torch.stack(terms).mean() if terms else logits.float().sum() * 0.0


class SentenceContextBroadcast(nn.Module):
    """Contextualize source sentences and inject them back into every token.

    Unlike a separate planner memory that the decoder can ignore, broadcasting
    the sentence representation changes the token memory consumed by every
    cross-attention layer.
    """

    def __init__(
        self,
        hidden_size: int,
        context_size: int,
        num_heads: int,
        ffn_size: int,
        num_layers: int,
        dropout: float,
        gate_init: float,
    ):
        super().__init__()
        if context_size % num_heads:
            raise ValueError("sentence context size must be divisible by its head count")
        if not 0.0 <= gate_init < 1.0:
            raise ValueError("sentence broadcast gate initialization must be in [0, 1)")
        self.input_norm = nn.RMSNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, context_size, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=context_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Pre-norm layers cannot use PyTorch's nested-tensor fast path. Disable
        # the attempted conversion explicitly to avoid a warning on every run.
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.RMSNorm(context_size)
        self.output_projection = nn.Linear(context_size, hidden_size, bias=False)
        raw_gate = math.atanh(gate_init)
        self.unit_gate = nn.Parameter(torch.tensor(raw_gate, dtype=torch.float32))
        self.token_gate = nn.Parameter(torch.tensor(raw_gate, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _position_encoding(length: int, width: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10_000.0) / max(1, width))
        )
        encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
        return encoding

    def forward(
        self,
        token_states: torch.Tensor,
        unit_ids: torch.Tensor,
        unit_count: int,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        units, valid_units = _pool_units(token_states, unit_ids, unit_count)
        if unit_count <= 0:
            return token_states, units, valid_units
        context = self.input_projection(self.input_norm(units))
        position = self._position_encoding(
            context.shape[1],
            context.shape[2],
            context.device,
        ).to(context.dtype)
        context = context + position.unsqueeze(0)
        context = self.encoder(context, src_key_padding_mask=~valid_units)
        context = context.masked_fill(~valid_units.unsqueeze(-1), 0)
        update = self.output_projection(self.output_norm(context))
        contextual_units = units + torch.tanh(self.unit_gate).to(units.dtype) * self.dropout(update)
        padded = torch.cat(
            [contextual_units.new_zeros(contextual_units.shape[0], 1, contextual_units.shape[2]), contextual_units],
            dim=1,
        )
        broadcast = padded.gather(
            1,
            unit_ids.clamp(min=0, max=unit_count).unsqueeze(-1).expand(-1, -1, token_states.shape[-1]),
        )
        token_states = token_states + torch.tanh(self.token_gate).to(token_states.dtype) * self.dropout(broadcast)
        token_states = token_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
        return token_states, contextual_units, valid_units


class _PlannerAttention(nn.Module):
    """Small SDPA attention used by the prospective-summary planner."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("planner hidden size must be divisible by its head count")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        key_value_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, query_length, _ = query_states.shape
        source_length = key_value_states.shape[1]
        query = self.q_proj(query_states).view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(key_value_states).view(batch, source_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(key_value_states).view(batch, source_length, self.num_heads, self.head_dim).transpose(1, 2)

        additive_mask: Optional[torch.Tensor] = None
        if key_value_mask is not None:
            valid = key_value_mask.bool()
            if valid.shape != (batch, source_length):
                raise ValueError(
                    f"planner source mask must have shape {(batch, source_length)}, got {tuple(valid.shape)}"
                )
            # SDPA cannot produce a meaningful result when every key is
            # masked.  Keep one zero-valued sentinel key for that rare case.
            no_source = ~valid.any(dim=-1)
            if bool(no_source.any()):
                valid = valid.clone()
                valid[no_source, 0] = True
                key = key.clone()
                value = value.clone()
                key[no_source, :, 0] = 0
                value[no_source, :, 0] = 0
            additive_mask = torch.zeros(
                (batch, 1, 1, source_length),
                device=query.device,
                dtype=query.dtype,
            )
            additive_mask.masked_fill_(~valid[:, None, None, :], torch.finfo(query.dtype).min)
        if attention_bias is not None:
            if attention_bias.shape != (batch, source_length):
                raise ValueError(
                    "planner attention bias must have shape "
                    f"{(batch, source_length)}, got {tuple(attention_bias.shape)}"
                )
            bias = attention_bias.to(device=query.device, dtype=query.dtype)[:, None, None, :]
            additive_mask = bias if additive_mask is None else additive_mask + bias

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=additive_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, query_length, self.hidden_size)
        return self.o_proj(attended)


class _ProspectiveSummaryBlock(nn.Module):
    """Ordered slot self-attention followed by slot-to-source attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        dropout: float,
        gate_init: float,
    ):
        super().__init__()
        if not 0.0 <= gate_init < 1.0:
            raise ValueError("planner gate initialization must be in [0, 1)")
        raw_gate = math.atanh(gate_init)
        self.self_norm = nn.RMSNorm(hidden_size)
        self.self_attention = _PlannerAttention(hidden_size, num_heads, dropout)
        self.self_gate = nn.Parameter(torch.tensor(raw_gate, dtype=torch.float32))
        self.cross_norm = nn.RMSNorm(hidden_size)
        self.memory_norm = nn.RMSNorm(hidden_size)
        self.cross_attention = _PlannerAttention(hidden_size, num_heads, dropout)
        self.cross_gate = nn.Parameter(torch.tensor(raw_gate, dtype=torch.float32))
        self.ffn_norm = nn.RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=False)
        self.ffn_gate = nn.Parameter(torch.tensor(raw_gate, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        slots: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self_update = self.self_attention(self.self_norm(slots), self.self_norm(slots))
        slots = slots + torch.tanh(self.self_gate).to(slots.dtype) * self.dropout(self_update)
        cross_update = self.cross_attention(
            self.cross_norm(slots),
            self.memory_norm(memory),
            key_value_mask=memory_mask,
            attention_bias=attention_bias,
        )
        slots = slots + torch.tanh(self.cross_gate).to(slots.dtype) * self.dropout(cross_update)
        normalized = self.ffn_norm(slots)
        ffn_update = self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        slots = slots + torch.tanh(self.ffn_gate).to(slots.dtype) * self.dropout(ffn_update)
        return slots


class ProspectiveSummaryPlanner(nn.Module):
    """Predict an ordered, fixed-length latent outline of the future summary.

    The slots are decoder-width vectors.  They interact with one another, then
    retrieve evidence from the *full* token memory; no source token is pooled
    away before retrieval.  Sentence salience is an optional additive bias,
    never a hard mask.
    """

    def __init__(
        self,
        hidden_size: int,
        num_slots: int = 16,
        num_layers: int = 2,
        num_heads: int = 16,
        ffn_size: Optional[int] = None,
        dropout: float = 0.1,
        gate_init: float = 0.1,
    ):
        super().__init__()
        if num_slots <= 0:
            raise ValueError("prospective summary planner requires at least one slot")
        if num_layers <= 0:
            raise ValueError("prospective summary planner requires at least one block")
        self.hidden_size = int(hidden_size)
        self.num_slots = int(num_slots)
        self.ordered_slots = nn.Parameter(torch.empty(self.num_slots, self.hidden_size))
        nn.init.normal_(self.ordered_slots, mean=0.0, std=0.02)
        width = int(ffn_size or self.hidden_size * 4)
        self.layers = nn.ModuleList(
            _ProspectiveSummaryBlock(
                self.hidden_size,
                int(num_heads),
                width,
                float(dropout),
                float(gate_init),
            )
            for _ in range(int(num_layers))
        )
        self.output_norm = nn.RMSNorm(self.hidden_size)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError(f"planner memory must be [B, S, D], got {tuple(memory.shape)}")
        if memory.shape[-1] != self.hidden_size:
            raise ValueError(f"planner expected memory width {self.hidden_size}, got {memory.shape[-1]}")
        slots = self.ordered_slots.to(memory.dtype).unsqueeze(0).expand(memory.shape[0], -1, -1)
        for layer in self.layers:
            slots = layer(slots, memory, memory_mask, attention_bias)
        return self.output_norm(slots)


@dataclass
class AdapterOutput:
    memory: torch.Tensor
    memory_mask: torch.Tensor
    attention_bias: Optional[torch.Tensor]
    salience_logits: Optional[torch.Tensor]
    loss_salience: torch.Tensor
    layer_weights: Optional[torch.Tensor]
    summary_prefix: Optional[torch.Tensor]
    summary_prefix_mask: Optional[torch.Tensor]


class ProspectiveSummaryBridge(nn.Module):
    """Raw encoder highway -> bidirectional memory -> ordered summary slots."""

    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        config: dict,
    ):
        super().__init__()
        self.encoder_hidden_size = int(encoder_hidden_size)
        self.decoder_hidden_size = int(decoder_hidden_size)
        self.hidden_size = int(config.get("hidden_size", decoder_hidden_size))
        self.use_layer_fusion = bool(config.get("layer_fusion", True))
        self.depth_routed_memory = bool(config.get("depth_routed_memory", False))
        self.layer_indices: List[int] = [int(value) for value in config.get("fuse_layers", [-1, -5, -9, -13])]
        dropout = float(config.get("dropout", 0.1))
        self.fusion = (
            StableTokenLayerFusion(
                self.encoder_hidden_size,
                self.layer_indices,
                dropout,
                float(config.get("layer_fusion_gate_init", 0.0)),
            )
            if self.use_layer_fusion
            else None
        )
        self.projection = IdentityResidualProjection(
            self.encoder_hidden_size,
            self.hidden_size,
            int(config.get("projection_ffn_size", self.hidden_size * 2)),
            dropout,
            float(config.get("projection_gate_init", 0.1)),
        )
        layer_count = int(config.get("num_bidirectional_layers", 4))
        self.bidirectional_layers = nn.ModuleList(
            BidirectionalAdapterLayer(
                self.hidden_size,
                int(config.get("num_heads", 16)),
                int(config.get("ffn_size", self.hidden_size * 4)),
                dropout,
                float(config.get("rope_theta", 1_000_000.0)),
                float(config.get("bidirectional_gate_init", 0.1)),
            )
            for _ in range(layer_count)
        )
        if self.depth_routed_memory:
            lexical_layers = [int(value) for value in config.get("lexical_layers", [-13, -9])]
            semantic_layers = [int(value) for value in config.get("semantic_layers", [-9, -5])]
            self.lexical_fusion: Optional[StableTokenLayerFusion] = StableTokenLayerFusion(
                self.encoder_hidden_size,
                lexical_layers,
                dropout,
                float(config.get("branch_layer_fusion_gate_init", 0.05)),
            )
            self.semantic_fusion: Optional[StableTokenLayerFusion] = StableTokenLayerFusion(
                self.encoder_hidden_size,
                semantic_layers,
                dropout,
                float(config.get("branch_layer_fusion_gate_init", 0.05)),
            )
            branch_ffn = int(config.get("branch_projection_ffn_size", self.hidden_size * 2))
            branch_gate = float(config.get("branch_projection_gate_init", 0.05))
            self.lexical_projection: Optional[IdentityResidualProjection] = IdentityResidualProjection(
                self.encoder_hidden_size,
                self.hidden_size,
                branch_ffn,
                dropout,
                branch_gate,
            )
            self.semantic_projection: Optional[IdentityResidualProjection] = IdentityResidualProjection(
                self.encoder_hidden_size,
                self.hidden_size,
                branch_ffn,
                dropout,
                branch_gate,
            )
            self.branch_global_context = bool(config.get("branch_global_context", False))
            if self.branch_global_context:
                raw_context_gate = math.atanh(float(config.get("branch_context_gate_init", 0.1)))
                self.lexical_context_gate = nn.Parameter(torch.tensor(raw_context_gate, dtype=torch.float32))
                self.semantic_context_gate = nn.Parameter(torch.tensor(raw_context_gate, dtype=torch.float32))
            else:
                self.register_parameter("lexical_context_gate", None)
                self.register_parameter("semantic_context_gate", None)
        else:
            self.lexical_fusion = None
            self.semantic_fusion = None
            self.lexical_projection = None
            self.semantic_projection = None
            self.branch_global_context = False
            self.register_parameter("lexical_context_gate", None)
            self.register_parameter("semantic_context_gate", None)
        self.use_sentence_context = bool(config.get("hierarchical_sentence_context", False))
        if self.use_sentence_context:
            self.sentence_context: Optional[SentenceContextBroadcast] = SentenceContextBroadcast(
                self.hidden_size,
                int(config.get("sentence_context_size", 256)),
                int(config.get("sentence_context_heads", 8)),
                int(config.get("sentence_context_ffn_size", 1024)),
                int(config.get("sentence_context_layers", 2)),
                dropout,
                float(config.get("sentence_broadcast_gate_init", 0.1)),
            )
        else:
            self.sentence_context = None
        if self.hidden_size == self.decoder_hidden_size:
            self.memory_projection: nn.Module = nn.Identity()
        else:
            self.memory_projection = nn.Sequential(
                nn.RMSNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.decoder_hidden_size, bias=False),
            )
            nn.init.xavier_uniform_(self.memory_projection[-1].weight)
        # Cross-attention already applies its copied decoder memory norm.  A
        # second unconditional RMSNorm here was one source of V3 coordinate
        # drift, so the identity-preserving V4 path leaves it disabled by
        # default.  It remains available only as an explicit compatibility
        # ablation.
        self.output_norm: nn.Module = (
            nn.RMSNorm(self.decoder_hidden_size) if bool(config.get("final_memory_norm", False)) else nn.Identity()
        )
        self.use_salience = bool(config.get("use_salience", True))
        if self.use_salience:
            salience_hidden = int(config.get("salience_hidden_size", max(128, self.hidden_size // 2)))
            self.salience_head = nn.Sequential(
                nn.RMSNorm(self.hidden_size),
                nn.Linear(self.hidden_size, salience_hidden, bias=False),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(salience_hidden, 1, bias=True),
            )
            self.salience_attention_gate = nn.Parameter(
                torch.tensor(math.atanh(float(config.get("salience_gate_init", 0.1))), dtype=torch.float32)
            )
        else:
            self.salience_head = None
            self.register_parameter("salience_attention_gate", None)
        self.salience_bias_scale = float(config.get("salience_bias_scale", 0.5))
        self._oracle_evidence_mix = 0.0

        self.use_summary_planner = bool(config.get("use_summary_planner", True))
        if self.use_summary_planner:
            planner_heads = int(config.get("summary_planner_heads", config.get("num_heads", 16)))
            self.summary_planner: Optional[ProspectiveSummaryPlanner] = ProspectiveSummaryPlanner(
                hidden_size=self.decoder_hidden_size,
                num_slots=int(config.get("num_summary_slots", 16)),
                num_layers=int(config.get("summary_planner_layers", 2)),
                num_heads=planner_heads,
                ffn_size=int(config.get("summary_planner_ffn_size", self.decoder_hidden_size * 4)),
                dropout=float(config.get("summary_planner_dropout", dropout)),
                gate_init=float(config.get("summary_planner_gate_init", 0.1)),
            )
        else:
            self.summary_planner = None

    @property
    def oracle_evidence_mix(self) -> float:
        return self._oracle_evidence_mix

    def set_oracle_evidence_mix(self, value: float) -> None:
        """Set the train-only teacher-forcing ratio for sentence evidence.

        Evaluation never consumes labels even if a stale training schedule left
        this value non-zero.  This makes validation/test leakage impossible by
        construction rather than by convention in the training loop.
        """

        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("oracle evidence mix must be in [0, 1]")
        self._oracle_evidence_mix = value

    def forward(
        self,
        encoder_hidden_states: Tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
        evidence_labels: Optional[torch.Tensor] = None,
    ) -> AdapterOutput:
        if self.fusion is not None:
            states = self.fusion(encoder_hidden_states, attention_mask)
        else:
            states = encoder_hidden_states[-1].masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
        states = self.projection(states)
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(~attention_mask.bool(), 0)
        for layer in self.bidirectional_layers:
            states = layer(states, attention_mask, position_ids)
        unit_count = 0
        contextual_units: Optional[torch.Tensor] = None
        valid_units: Optional[torch.Tensor] = None
        if unit_ids is not None:
            observed_units = int(unit_ids.max().item()) if unit_ids.numel() else 0
            labelled_units = int(evidence_labels.shape[1]) if evidence_labels is not None else 0
            unit_count = max(observed_units, labelled_units)
            if self.sentence_context is not None:
                states, contextual_units, valid_units = self.sentence_context(
                    states,
                    unit_ids,
                    unit_count,
                    attention_mask,
                )
        summary_memory = self.output_norm(self.memory_projection(states))
        summary_memory = summary_memory.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)

        zero = summary_memory.float().sum() * 0.0
        salience_logits: Optional[torch.Tensor] = None
        attention_bias: Optional[torch.Tensor] = None
        loss_salience = zero
        if self.salience_head is not None and unit_ids is not None:
            if contextual_units is None or valid_units is None:
                units, valid_units = _pool_units(states, unit_ids, unit_count)
            else:
                units = contextual_units
            salience_logits = self.salience_head(units).squeeze(-1)
            labels: Optional[torch.Tensor] = None
            if evidence_labels is not None:
                labels = evidence_labels
                if labels.shape[1] < salience_logits.shape[1]:
                    labels = F.pad(labels, (0, salience_logits.shape[1] - labels.shape[1]), value=-1)
                labels = labels[:, : salience_logits.shape[1]]
                loss_salience = _balanced_salience_loss(salience_logits, labels, valid_units)

            predicted_probability = torch.sigmoid(salience_logits.float())
            mixed_probability = predicted_probability
            # Oracle evidence is a training curriculum only.  In eval mode the
            # prediction path is used unconditionally, even if labels happen to
            # be present in the batch.
            if self.training and labels is not None and self._oracle_evidence_mix > 0.0:
                supervised = valid_units & labels.ge(0)
                oracle_probability = labels.float().clamp(0.0, 1.0)
                scheduled = torch.lerp(
                    predicted_probability,
                    oracle_probability,
                    self._oracle_evidence_mix,
                )
                mixed_probability = torch.where(supervised, scheduled, predicted_probability)

            # A centred log-probability supplies both positive and negative
            # evidence while keeping finite values for hard oracle labels.
            unit_bias = torch.logit(mixed_probability.clamp(0.02, 0.98))
            padded_bias = torch.cat(
                [unit_bias.new_zeros(unit_bias.shape[0], 1), unit_bias],
                dim=1,
            )
            gathered = padded_bias.gather(1, unit_ids.clamp(min=0, max=unit_count))
            bias_gate = torch.tanh(self.salience_attention_gate.float()).to(gathered.dtype)
            attention_bias = bias_gate * self.salience_bias_scale * gathered
            attention_bias = attention_bias.to(summary_memory.dtype)
            attention_bias = attention_bias.masked_fill(~attention_mask.bool(), 0)

        summary_prefix: Optional[torch.Tensor]
        summary_prefix_mask: Optional[torch.Tensor]
        if self.summary_planner is not None:
            summary_prefix = self.summary_planner(
                summary_memory,
                attention_mask,
                attention_bias,
            )
            summary_prefix_mask = torch.ones(
                summary_prefix.shape[:2],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        else:
            summary_prefix = None
            summary_prefix_mask = None

        if self.depth_routed_memory:
            assert self.lexical_fusion is not None
            assert self.semantic_fusion is not None
            assert self.lexical_projection is not None
            assert self.semantic_projection is not None
            lexical = self.lexical_projection(self.lexical_fusion(encoder_hidden_states, attention_mask))
            semantic = self.semantic_projection(self.semantic_fusion(encoder_hidden_states, attention_mask))
            if self.branch_global_context:
                assert self.lexical_context_gate is not None
                assert self.semantic_context_gate is not None
                lexical = lexical + torch.tanh(self.lexical_context_gate.float()).to(lexical.dtype) * states
                semantic = semantic + torch.tanh(self.semantic_context_gate.float()).to(semantic.dtype) * states
            lexical_memory = self.output_norm(self.memory_projection(lexical))
            semantic_memory = self.output_norm(self.memory_projection(semantic))
            lexical_memory = lexical_memory.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
            semantic_memory = semantic_memory.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
            memory = torch.stack(
                [lexical_memory, semantic_memory, summary_memory],
                dim=1,
            )
        else:
            memory = summary_memory
        return AdapterOutput(
            memory=memory,
            memory_mask=attention_mask,
            attention_bias=attention_bias,
            salience_logits=salience_logits,
            loss_salience=loss_salience,
            layer_weights=self.fusion.last_mean_weights if self.fusion is not None else None,
            summary_prefix=summary_prefix,
            summary_prefix_mask=summary_prefix_mask,
        )

    def bidirectional_gate_mean(self) -> torch.Tensor:
        values = []
        for layer in self.bidirectional_layers:
            values.extend(
                [
                    torch.tanh(layer.attention_gate.float()),
                    torch.tanh(layer.ffn_gate.float()),
                ]
            )
        if not values:
            return self.projection.residual_gate.new_zeros(())
        return torch.stack(values).mean()

    def branch_context_gate_mean(self) -> torch.Tensor:
        if self.lexical_context_gate is None or self.semantic_context_gate is None:
            return self.projection.residual_gate.new_zeros(())
        return torch.stack(
            [
                torch.tanh(self.lexical_context_gate.float()),
                torch.tanh(self.semantic_context_gate.float()),
            ]
        ).mean()


# Compatibility alias for tensor-only tests and any early V4 pilot configs.
# New code and paper descriptions use the architecture's actual V4 name.
SummaryAdapterV2 = ProspectiveSummaryBridge
