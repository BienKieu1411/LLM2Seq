"""Summary-specific hierarchical bidirectional evidence bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BridgeOutput:
    memory: torch.Tensor
    memory_mask: torch.Tensor
    salience_logits: Optional[torch.Tensor]
    loss_evidence: torch.Tensor
    loss_diversity: torch.Tensor


def _unit_pool(
    token_states: torch.Tensor,
    unit_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool ids 1..S; id 0 denotes prompt/padding and is excluded."""

    batch, _, hidden = token_states.shape
    unit_count = int(unit_ids.max().item()) if unit_ids.numel() else 0
    if unit_count == 0:
        # Keep a masked dummy position so Transformer/MHA shapes remain valid.
        states = token_states.new_zeros((batch, 1, hidden))
        valid = torch.zeros((batch, 1), dtype=torch.bool, device=token_states.device)
        return states, valid
    pooled = token_states.new_zeros((batch, unit_count + 1, hidden))
    counts = token_states.new_zeros((batch, unit_count + 1, 1))
    indices = unit_ids.clamp(min=0, max=unit_count).unsqueeze(-1).expand(-1, -1, hidden)
    pooled.scatter_add_(1, indices, token_states)
    counts.scatter_add_(
        1,
        unit_ids.clamp(min=0, max=unit_count).unsqueeze(-1),
        torch.ones_like(token_states[..., :1]),
    )
    pooled = pooled[:, 1:] / counts[:, 1:].clamp_min(1.0)
    valid = counts[:, 1:, 0].gt(0)
    return pooled, valid


def _gather_unit_context(
    unit_states: torch.Tensor,
    unit_ids: torch.Tensor,
) -> torch.Tensor:
    padded = torch.cat([unit_states.new_zeros(unit_states.shape[0], 1, unit_states.shape[-1]), unit_states], dim=1)
    indices = unit_ids.clamp(min=0, max=unit_states.shape[1]).unsqueeze(-1).expand(-1, -1, unit_states.shape[-1])
    return padded.gather(1, indices)


class EvidenceBridge(nn.Module):
    """Plan salient source content without altering the causal LLM backbone.

    The bridge performs bidirectional reasoning only over evidence-unit vectors,
    predicts reference-derived evidence labels, and creates prompt-conditioned
    evidence slots. The decoder receives both the complete token memory and the
    compact slots, so evidence selection cannot irreversibly delete source facts.
    """

    def __init__(
        self,
        encoder_size: int,
        decoder_size: int,
        config: Dict[str, object],
    ):
        super().__init__()
        self.mode = str(config.get("mode", "evidence"))
        if self.mode not in {"causal", "lamate", "hierarchical", "evidence", "slots_only"}:
            raise ValueError(f"Unknown bridge mode: {self.mode}")
        bridge_size = int(config.get("hidden_size", min(encoder_size, 768)))
        num_heads = int(config.get("num_heads", 8))
        if bridge_size % num_heads:
            raise ValueError("bridge.hidden_size must be divisible by bridge.num_heads")
        self.encoder_size = int(encoder_size)
        self.decoder_size = int(decoder_size)
        self.bridge_size = bridge_size
        self.num_slots = int(config.get("num_evidence_slots", 16)) if self.mode in {"evidence", "slots_only"} else 0
        self.use_evidence_loss = bool(config.get("use_evidence_loss", True)) and self.mode in {"evidence", "slots_only"}
        dropout = float(config.get("dropout", 0.1))

        self.token_norm = nn.LayerNorm(encoder_size)
        self.token_to_decoder = nn.Linear(encoder_size, decoder_size, bias=False)
        self.token_to_bridge = nn.Sequential(
            nn.Linear(encoder_size, bridge_size, bias=False),
            nn.GELU(),
            nn.Linear(bridge_size, bridge_size, bias=False),
        )
        self.anchor_projection = nn.Linear(encoder_size, bridge_size, bias=False)

        self.token_postencoder = None
        if self.mode == "causal":
            self.unit_encoder = None
            self.salience_head = None
            self.slot_attention = None
            self.register_parameter("slot_queries", None)
            self.register_parameter("broadcast_gate", None)
        elif self.mode == "lamate":
            # Width/depth-matched external bidirectional token encoder. This is
            # the controlled LaMaTE-style baseline, not the proposed method.
            layer = nn.TransformerEncoderLayer(
                d_model=bridge_size,
                nhead=num_heads,
                dim_feedforward=int(config.get("ffn_size", 4 * bridge_size)),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_postencoder = nn.TransformerEncoder(
                layer,
                num_layers=int(config.get("num_layers", 3)),
                norm=nn.LayerNorm(bridge_size),
                enable_nested_tensor=False,
            )
            self.token_post_to_decoder = nn.Linear(bridge_size, decoder_size, bias=False)
            self.unit_encoder = None
            self.salience_head = None
            self.slot_attention = None
            self.register_parameter("slot_queries", None)
            self.register_parameter("broadcast_gate", None)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=bridge_size,
                nhead=num_heads,
                dim_feedforward=int(config.get("ffn_size", 4 * bridge_size)),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.unit_encoder = nn.TransformerEncoder(
                layer,
                num_layers=int(config.get("num_layers", 3)),
                norm=nn.LayerNorm(bridge_size),
                enable_nested_tensor=False,
            )
            self.salience_head = nn.Sequential(
                nn.LayerNorm(bridge_size),
                nn.Linear(bridge_size, bridge_size // 2),
                nn.GELU(),
                nn.Linear(bridge_size // 2, 1),
            )
            self.broadcast_projection = nn.Linear(bridge_size, decoder_size, bias=False)
            self.broadcast_gate = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
            if self.num_slots:
                self.slot_queries = nn.Parameter(torch.empty(self.num_slots, bridge_size))
                nn.init.normal_(self.slot_queries, std=0.02)
                self.slot_attention = nn.MultiheadAttention(
                    bridge_size,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.slot_norm = nn.LayerNorm(bridge_size)
                self.slot_to_decoder = nn.Linear(bridge_size, decoder_size, bias=False)
            else:
                self.slot_attention = None
                self.register_parameter("slot_queries", None)

        self.output_norm = nn.LayerNorm(decoder_size)

    @staticmethod
    def _last_valid(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        indices = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), indices]

    @staticmethod
    def _diversity_loss(slots: torch.Tensor) -> torch.Tensor:
        if slots.shape[1] <= 1:
            return slots.new_zeros(())
        normalized = F.normalize(slots.float(), dim=-1)
        similarity = normalized @ normalized.transpose(1, 2)
        identity = torch.eye(slots.shape[1], device=slots.device, dtype=torch.bool)[None]
        return similarity.masked_select(~identity).square().mean().to(slots.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_ids: torch.Tensor,
        evidence_labels: Optional[torch.Tensor] = None,
    ) -> BridgeOutput:
        normalized = self.token_norm(hidden_states)
        token_memory = self.token_to_decoder(normalized)
        zero = hidden_states.new_zeros(())
        if self.mode == "causal":
            return BridgeOutput(self.output_norm(token_memory), attention_mask, None, zero, zero)

        bridge_tokens = self.token_to_bridge(normalized)
        if self.mode == "lamate":
            contextual = self.token_postencoder(
                bridge_tokens,
                src_key_padding_mask=~attention_mask.bool(),
            )
            contextual = self.token_post_to_decoder(contextual)
            contextual = contextual.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
            return BridgeOutput(self.output_norm(contextual), attention_mask, None, zero, zero)

        unit_states, unit_mask = _unit_pool(bridge_tokens, unit_ids)
        unit_states = self.unit_encoder(
            unit_states,
            src_key_padding_mask=~unit_mask,
        )
        unit_states = unit_states.masked_fill(~unit_mask.unsqueeze(-1), 0)
        salience_logits = self.salience_head(unit_states).squeeze(-1)

        unit_context = _gather_unit_context(unit_states, unit_ids)
        broadcast = self.broadcast_projection(unit_context)
        token_memory = token_memory + torch.tanh(self.broadcast_gate).to(token_memory.dtype) * broadcast

        loss_evidence = zero
        if evidence_labels is not None and self.use_evidence_loss:
            width = min(evidence_labels.shape[1], salience_logits.shape[1])
            labels = evidence_labels[:, :width]
            valid = labels.ge(0) & unit_mask[:, :width]
            if bool(valid.any()):
                loss_evidence = F.binary_cross_entropy_with_logits(
                    salience_logits[:, :width][valid].float(),
                    labels[valid].float(),
                ).to(hidden_states.dtype)

        slots = None
        loss_diversity = zero
        if self.slot_attention is not None:
            anchor = self.anchor_projection(self._last_valid(normalized, attention_mask))
            queries = self.slot_queries.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
            queries = queries + anchor.unsqueeze(1)
            # Salience is a soft prior, not a hard selector. This lets summary CE
            # train planning even in the no-evidence-supervision ablation.
            weighted_units = unit_states * (0.5 + torch.sigmoid(salience_logits)).unsqueeze(-1)
            slots, _ = self.slot_attention(
                queries,
                weighted_units,
                weighted_units,
                key_padding_mask=~unit_mask,
                need_weights=False,
            )
            slots = self.slot_norm(slots + queries)
            loss_diversity = self._diversity_loss(slots)
            slots = self.slot_to_decoder(slots)

        token_memory = self.output_norm(token_memory)
        if self.mode == "slots_only":
            if slots is None:
                raise RuntimeError("slots_only requires num_evidence_slots > 0")
            slot_mask = torch.ones(slots.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
            return BridgeOutput(self.output_norm(slots), slot_mask, salience_logits, loss_evidence, loss_diversity)
        if slots is None:
            return BridgeOutput(token_memory, attention_mask, salience_logits, loss_evidence, loss_diversity)
        slot_mask = torch.ones(slots.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
        memory = torch.cat([self.output_norm(slots), token_memory], dim=1)
        memory_mask = torch.cat([slot_mask, attention_mask], dim=1)
        return BridgeOutput(memory, memory_mask, salience_logits, loss_evidence, loss_diversity)
