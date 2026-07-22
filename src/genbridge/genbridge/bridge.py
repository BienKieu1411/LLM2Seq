"""Bidirectional, output-oriented adapter specialized for summarization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class BidirectionalRotarySelfAttention(nn.Module):
    """Full self-attention with Qwen-style RoPE and no causal mask.

    Causal Qwen states already contain contextual order information, but a
    newly projected post-encoder otherwise has no direct relative-position
    mechanism of its own.  RoPE makes the adapter explicitly order-aware
    without adding learned position parameters or modifying the Qwen backbone.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        rope_theta: float,
        use_rope: bool,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        self.dropout = float(dropout)
        self.rope_theta = float(rope_theta)
        self.use_rope = bool(use_rope)
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # Match MultiheadAttention's stable QKV initialization instead of the
        # generic Linear Kaiming default.
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inverse_frequency = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    device=query.device,
                    dtype=torch.float32,
                )
                / self.head_dim
            )
        )
        frequency = position_ids.float().unsqueeze(-1) * inverse_frequency
        embedding = torch.cat([frequency, frequency], dim=-1)
        cosine = embedding.cos().to(query.dtype).unsqueeze(1)
        sine = embedding.sin().to(query.dtype).unsqueeze(1)
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        query = query.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        if self.use_rope:
            query, key = self._apply_rope(query, key, position_ids)
        # PyTorch SDPA boolean masks use True for entries that participate in
        # attention. There is deliberately no triangular/causal component.
        key_mask = attention_mask.bool()[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=key_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.hidden_size
        )
        attended = self.out_proj(attended)
        return attended.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)


class BidirectionalRotaryEncoderLayer(nn.Module):
    """Pre-norm bidirectional adapter block with an explicit order signal."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        dropout: float,
        rope_theta: float,
        use_rope: bool,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = BidirectionalRotarySelfAttention(
            hidden_size,
            num_heads,
            dropout,
            rope_theta,
            use_rope,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn_in = nn.Linear(hidden_size, ffn_size)
        self.ffn_activation_dropout = nn.Dropout(dropout)
        self.ffn_out = nn.Linear(ffn_size, hidden_size)
        self.ffn_output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.attention(
            self.attention_norm(hidden_states),
            attention_mask,
            position_ids,
        )
        hidden_states = hidden_states + self.attention_dropout(attention_output)
        ffn_output = self.ffn_in(self.ffn_norm(hidden_states))
        ffn_output = self.ffn_activation_dropout(F.gelu(ffn_output))
        ffn_output = self.ffn_out(ffn_output)
        hidden_states = hidden_states + self.ffn_output_dropout(ffn_output)
        return hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)


class BidirectionalRotaryEncoder(nn.Module):
    """Stack adapter layers using compact positions invariant to left padding."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        dropout: float,
        num_layers: int,
        rope_theta: float,
        use_rope: bool,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("Bidirectional adapter must contain at least one layer")
        # Construct independently: deepcopy-cloned TransformerEncoder layers
        # start with identical parameters and waste scarce task supervision
        # breaking that symmetry during adapter warm-up.
        self.layers = nn.ModuleList(
            BidirectionalRotaryEncoderLayer(
                hidden_size,
                num_heads,
                ffn_size,
                dropout,
                rope_theta,
                use_rope,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.use_rope = bool(use_rope)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask = attention_mask.bool()
        if bool((~attention_mask.any(dim=1)).any()):
            raise ValueError("Every adapter sequence must contain at least one valid token")
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(~attention_mask, 0)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
        hidden_states = self.output_norm(hidden_states)
        return hidden_states.masked_fill(~attention_mask.unsqueeze(-1), 0)


@dataclass
class BridgeOutput:
    # ``memory`` is kept as the concatenated/single-memory representation for
    # compatibility and for the controlled concatenation ablation.  The main
    # model passes the two branches below separately to the decoder.
    memory: torch.Tensor
    memory_mask: torch.Tensor
    token_memory: torch.Tensor
    token_memory_mask: torch.Tensor
    plan_memory: torch.Tensor
    plan_memory_mask: torch.Tensor
    salience_logits: Optional[torch.Tensor]
    loss_salience: torch.Tensor
    loss_plan_diversity: torch.Tensor


def _unit_pool(
    token_states: torch.Tensor,
    unit_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool ids 1..S; id 0 denotes prompt/padding."""

    batch, _, hidden = token_states.shape
    unit_count = int(unit_ids.max().item()) if unit_ids.numel() else 0
    if unit_count == 0:
        states = token_states.new_zeros((batch, 1, hidden))
        valid = torch.zeros((batch, 1), dtype=torch.bool, device=token_states.device)
        # A valid dummy avoids all-masked attention NaNs. It is never supervised.
        valid[:, 0] = True
        return states, valid
    pooled = token_states.new_zeros((batch, unit_count + 1, hidden))
    counts = token_states.new_zeros((batch, unit_count + 1, 1))
    clamped = unit_ids.clamp(min=0, max=unit_count)
    indices = clamped.unsqueeze(-1).expand(-1, -1, hidden)
    pooled.scatter_add_(1, indices, token_states)
    counts.scatter_add_(
        1,
        clamped.unsqueeze(-1),
        torch.ones_like(token_states[..., :1]),
    )
    pooled = pooled[:, 1:] / counts[:, 1:].clamp_min(1.0)
    valid = counts[:, 1:, 0].gt(0)
    return pooled, valid


def _gather_unit_context(unit_states: torch.Tensor, unit_ids: torch.Tensor) -> torch.Tensor:
    padding = unit_states.new_zeros(unit_states.shape[0], 1, unit_states.shape[-1])
    padded = torch.cat([padding, unit_states], dim=1)
    indices = unit_ids.clamp(min=0, max=unit_states.shape[1]).unsqueeze(-1)
    return padded.gather(1, indices.expand(-1, -1, unit_states.shape[-1]))


def _balanced_binary_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Give positive and negative evidence units equal aggregate weight.

    WikiLingua has many more non-evidence than evidence sentences. Averaging
    ordinary BCE over all units therefore permits a low-loss all-negative
    salience predictor. Averaging the two classes separately keeps the loss
    scale unchanged while preventing that collapse.
    """

    positive = valid & labels.gt(0.5)
    negative = valid & labels.le(0.5)
    terms = []
    if bool(positive.any()):
        terms.append(F.softplus(-logits[positive].float()).mean())
    if bool(negative.any()):
        terms.append(F.softplus(logits[negative].float()).mean())
    if not terms:
        return logits.new_zeros(())
    return torch.stack(terms).mean().to(logits.dtype)


class SummaryBridge(nn.Module):
    """Turn causal Qwen states into bidirectional token and summary-plan memory.

    Main mode (``genbridge``) has three summary-specific mechanisms:

    1. a full bidirectional post-encoder over token states;
    2. reference-supervised sentence salience with soft broadcast to tokens;
    3. causal suffix planning states refined over salient sentence memory.

    The complete token memory is retained, so the small plan is guidance rather
    than an irreversible information bottleneck.
    """

    MODES = {"causal", "lamate", "hierarchical", "genbridge", "plan_only"}

    def __init__(self, encoder_size: int, decoder_size: int, config: Dict[str, object]):
        super().__init__()
        self.mode = str(config.get("mode", "genbridge"))
        if self.mode not in self.MODES:
            raise ValueError(f"Unknown bridge mode: {self.mode}")

        bridge_size = int(config.get("hidden_size", min(encoder_size, 512)))
        num_heads = int(config.get("num_heads", 8))
        if bridge_size % num_heads:
            raise ValueError("bridge.hidden_size must be divisible by bridge.num_heads")
        dropout = float(config.get("dropout", 0.1))
        ffn_size = int(config.get("ffn_size", 4 * bridge_size))
        rope_theta = float(config.get("rope_theta", 1_000_000.0))
        use_adapter_rope = bool(config.get("use_adapter_rope", True))
        self.encoder_size = int(encoder_size)
        self.decoder_size = int(decoder_size)
        self.bridge_size = bridge_size
        self.use_salience_loss = bool(config.get("use_salience_loss", True))
        self.use_plan_alignment = bool(config.get("use_plan_alignment", True))
        self.balance_salience_loss = bool(config.get("balance_salience_loss", True))

        self.input_norm = nn.LayerNorm(encoder_size)
        self.token_in = nn.Linear(encoder_size, bridge_size, bias=False)
        self.plan_in = nn.Linear(encoder_size, bridge_size, bias=False)

        self.token_encoder: Optional[nn.Module] = None
        if self.mode != "causal":
            self.token_encoder = BidirectionalRotaryEncoder(
                hidden_size=bridge_size,
                num_heads=num_heads,
                ffn_size=ffn_size,
                dropout=dropout,
                num_layers=int(config.get("token_num_layers", 4)),
                rope_theta=rope_theta,
                use_rope=use_adapter_rope,
            )

        self.unit_encoder: Optional[nn.Module] = None
        self.salience_head: Optional[nn.Module] = None
        self.plan_attention: Optional[nn.MultiheadAttention] = None
        if self.mode in {"hierarchical", "genbridge", "plan_only"}:
            self.unit_encoder = BidirectionalRotaryEncoder(
                hidden_size=bridge_size,
                num_heads=num_heads,
                ffn_size=ffn_size,
                dropout=dropout,
                num_layers=int(config.get("unit_num_layers", 1)),
                rope_theta=rope_theta,
                use_rope=use_adapter_rope,
            )
            self.salience_head = nn.Sequential(
                nn.LayerNorm(bridge_size),
                nn.Linear(bridge_size, max(64, bridge_size // 2)),
                nn.GELU(),
                nn.Linear(max(64, bridge_size // 2), 1),
            )
            self.plan_attention = nn.MultiheadAttention(
                bridge_size,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.plan_norm = nn.LayerNorm(bridge_size)
            self.unit_to_token = nn.Linear(bridge_size, bridge_size, bias=False)
            # Zero-like initialization preserves the token adapter at startup.
            self.unit_broadcast_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.token_out = nn.Linear(bridge_size, decoder_size, bias=False)
        self.plan_out = nn.Linear(bridge_size, decoder_size, bias=False)
        # Preserve native Qwen coordinates so cross-attention copied from
        # Qwen self-attention receives a familiar K/V distribution. The
        # bidirectional adapter begins as a small residual correction instead
        # of replacing every source state with a random projection.
        if encoder_size == decoder_size:
            self.token_skip: nn.Module = nn.Identity()
            self.plan_skip: nn.Module = nn.Identity()
            # The Qwen backbone has already applied its learned final RMSNorm.
            # A fresh LayerNorm here would no longer be an identity-preserving
            # bridge (and changes the scale/direction seen by copied K/V).
            self.token_output_norm: nn.Module = nn.Identity()
            self.plan_output_norm: nn.Module = nn.Identity()
        else:
            self.token_skip = nn.Linear(encoder_size, decoder_size, bias=False)
            self.plan_skip = nn.Linear(encoder_size, decoder_size, bias=False)
            nn.init.xavier_uniform_(self.token_skip.weight)
            nn.init.xavier_uniform_(self.plan_skip.weight)
            self.token_output_norm = nn.LayerNorm(decoder_size)
            self.plan_output_norm = nn.LayerNorm(decoder_size)
        token_adapter_gate_init = float(config.get("token_adapter_gate_init", 0.1))
        if not 0.0 < token_adapter_gate_init < 1.0:
            raise ValueError("bridge.token_adapter_gate_init must be in (0, 1)")
        self.token_adapter_gate = nn.Parameter(
            torch.tensor(math.atanh(token_adapter_gate_init), dtype=torch.float32)
        )
        plan_adapter_gate_init = float(config.get("plan_adapter_gate_init", 0.1))
        if not 0.0 < plan_adapter_gate_init < 1.0:
            raise ValueError("bridge.plan_adapter_gate_init must be in (0, 1)")
        self.plan_adapter_gate = nn.Parameter(
            torch.tensor(math.atanh(plan_adapter_gate_init), dtype=torch.float32)
        )
        self.token_type = nn.Parameter(torch.zeros(decoder_size))
        self.plan_type = nn.Parameter(torch.zeros(decoder_size))

    @staticmethod
    def _diversity_loss(plans: torch.Tensor) -> torch.Tensor:
        if plans.shape[1] <= 1:
            return plans.new_zeros(())
        normalized = F.normalize(plans.float(), dim=-1)
        similarity = normalized @ normalized.transpose(1, 2)
        identity = torch.eye(plans.shape[1], device=plans.device, dtype=torch.bool)[None]
        return similarity.masked_select(~identity).square().mean().to(plans.dtype)

    def forward(
        self,
        token_states: torch.Tensor,
        plan_states: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_ids: torch.Tensor,
        evidence_labels: Optional[torch.Tensor] = None,
    ) -> BridgeOutput:
        token_hidden = self.token_in(self.input_norm(token_states))
        plan_hidden = self.plan_in(self.input_norm(plan_states))
        zero = token_states.new_zeros(())

        # No causal mask is passed here: this adapter is explicitly
        # bidirectional while the pretrained Qwen backbone remains untouched.
        if self.token_encoder is not None:
            token_hidden = self.token_encoder(
                token_hidden,
                attention_mask,
            )
        token_hidden = token_hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)

        salience_logits: Optional[torch.Tensor] = None
        loss_salience = zero
        if self.unit_encoder is not None and self.salience_head is not None:
            unit_states, unit_mask = _unit_pool(token_hidden, unit_ids)
            unit_states = self.unit_encoder(
                unit_states,
                unit_mask,
            )
            unit_states = unit_states.masked_fill(~unit_mask.unsqueeze(-1), 0)
            salience_logits = self.salience_head(unit_states).squeeze(-1)
            if evidence_labels is not None and self.use_salience_loss:
                width = min(evidence_labels.shape[1], salience_logits.shape[1])
                labels = evidence_labels[:, :width]
                valid = labels.ge(0) & unit_mask[:, :width]
                if bool(valid.any()):
                    selected_logits = salience_logits[:, :width]
                    if self.balance_salience_loss:
                        loss_salience = _balanced_binary_cross_entropy(
                            selected_logits,
                            labels,
                            valid,
                        ).to(token_states.dtype)
                    else:
                        loss_salience = F.binary_cross_entropy_with_logits(
                            selected_logits[valid].float(),
                            labels[valid].float(),
                        ).to(token_states.dtype)

            unit_context = _gather_unit_context(unit_states, unit_ids)
            token_hidden = token_hidden + torch.tanh(self.unit_broadcast_gate).to(
                token_hidden.dtype
            ) * self.unit_to_token(unit_context)

            if self.mode in {"genbridge", "plan_only"}:
                weighted_values = unit_states * (
                    0.5 + torch.sigmoid(salience_logits)
                ).unsqueeze(-1)
                planned, _ = self.plan_attention(
                    plan_hidden,
                    # Keep semantic addressing in the keys stable; salience
                    # controls how much evidence each selected value carries.
                    unit_states,
                    weighted_values,
                    key_padding_mask=~unit_mask,
                    need_weights=False,
                )
                plan_hidden = self.plan_norm(plan_hidden + planned)

        plan_adapter_gate = torch.tanh(self.plan_adapter_gate).to(plan_hidden.dtype)
        plan_memory = self.plan_output_norm(
            self.plan_skip(plan_states)
            + plan_adapter_gate * self.plan_out(plan_hidden)
            + self.plan_type
        )
        adapter_gate = torch.tanh(self.token_adapter_gate).to(token_hidden.dtype)
        token_memory = self.token_output_norm(
            self.token_skip(token_states)
            + adapter_gate * self.token_out(token_hidden)
            + self.token_type
        )
        plan_mask = torch.ones(
            plan_memory.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        loss_plan_diversity = self._diversity_loss(plan_memory)

        if self.mode == "plan_only":
            memory, memory_mask = plan_memory, plan_mask
        elif self.mode in {"genbridge"}:
            memory = torch.cat([plan_memory, token_memory], dim=1)
            memory_mask = torch.cat([plan_mask, attention_mask], dim=1)
        else:
            # causal, lamate and hierarchical isolate token-memory effects.
            memory, memory_mask = token_memory, attention_mask

        return BridgeOutput(
            memory=memory,
            memory_mask=memory_mask,
            token_memory=token_memory,
            token_memory_mask=attention_mask,
            plan_memory=plan_memory,
            plan_memory_mask=plan_mask,
            salience_logits=salience_logits,
            loss_salience=loss_salience,
            loss_plan_diversity=loss_plan_diversity,
        )


# Backward-compatible import name used by the original test scaffold.
EvidenceBridge = SummaryBridge
