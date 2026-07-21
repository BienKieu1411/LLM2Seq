"""Bidirectional, output-oriented adapter specialized for summarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.encoder_size = int(encoder_size)
        self.decoder_size = int(decoder_size)
        self.bridge_size = bridge_size
        self.use_salience_loss = bool(config.get("use_salience_loss", True))
        self.use_plan_alignment = bool(config.get("use_plan_alignment", True))

        self.input_norm = nn.LayerNorm(encoder_size)
        self.token_in = nn.Linear(encoder_size, bridge_size, bias=False)
        self.plan_in = nn.Linear(encoder_size, bridge_size, bias=False)

        self.token_encoder: Optional[nn.Module] = None
        if self.mode != "causal":
            token_layer = nn.TransformerEncoderLayer(
                d_model=bridge_size,
                nhead=num_heads,
                dim_feedforward=ffn_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_encoder = nn.TransformerEncoder(
                token_layer,
                num_layers=int(config.get("token_num_layers", 2)),
                norm=nn.LayerNorm(bridge_size),
                enable_nested_tensor=False,
            )

        self.unit_encoder: Optional[nn.Module] = None
        self.salience_head: Optional[nn.Module] = None
        self.plan_attention: Optional[nn.MultiheadAttention] = None
        if self.mode in {"hierarchical", "genbridge", "plan_only"}:
            unit_layer = nn.TransformerEncoderLayer(
                d_model=bridge_size,
                nhead=num_heads,
                dim_feedforward=ffn_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.unit_encoder = nn.TransformerEncoder(
                unit_layer,
                num_layers=int(config.get("unit_num_layers", 1)),
                norm=nn.LayerNorm(bridge_size),
                enable_nested_tensor=False,
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
        self.token_type = nn.Parameter(torch.zeros(decoder_size))
        self.plan_type = nn.Parameter(torch.zeros(decoder_size))
        self.output_norm = nn.LayerNorm(decoder_size)

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
                src_key_padding_mask=~attention_mask.bool(),
            )
        token_hidden = token_hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)

        salience_logits: Optional[torch.Tensor] = None
        loss_salience = zero
        if self.unit_encoder is not None and self.salience_head is not None:
            unit_states, unit_mask = _unit_pool(token_hidden, unit_ids)
            unit_states = self.unit_encoder(
                unit_states,
                src_key_padding_mask=~unit_mask,
            )
            unit_states = unit_states.masked_fill(~unit_mask.unsqueeze(-1), 0)
            salience_logits = self.salience_head(unit_states).squeeze(-1)
            if evidence_labels is not None and self.use_salience_loss:
                width = min(evidence_labels.shape[1], salience_logits.shape[1])
                labels = evidence_labels[:, :width]
                valid = labels.ge(0) & unit_mask[:, :width]
                if bool(valid.any()):
                    loss_salience = F.binary_cross_entropy_with_logits(
                        salience_logits[:, :width][valid].float(),
                        labels[valid].float(),
                    ).to(token_states.dtype)

            unit_context = _gather_unit_context(unit_states, unit_ids)
            token_hidden = token_hidden + torch.tanh(self.unit_broadcast_gate).to(
                token_hidden.dtype
            ) * self.unit_to_token(unit_context)

            if self.mode in {"genbridge", "plan_only"}:
                weighted_units = unit_states * (
                    0.5 + torch.sigmoid(salience_logits)
                ).unsqueeze(-1)
                planned, _ = self.plan_attention(
                    plan_hidden,
                    weighted_units,
                    weighted_units,
                    key_padding_mask=~unit_mask,
                    need_weights=False,
                )
                plan_hidden = self.plan_norm(plan_hidden + planned)

        plan_memory = self.output_norm(self.plan_out(plan_hidden) + self.plan_type)
        token_memory = self.output_norm(self.token_out(token_hidden) + self.token_type)
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
