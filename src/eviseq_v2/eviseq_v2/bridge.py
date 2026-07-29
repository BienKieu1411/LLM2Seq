"""Single-memory evidence bridge; no planner, HiRoute, or scratch transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .native_attention import unit_evidence_token_bias


@dataclass
class BridgeOutput:
    memory: torch.Tensor
    memory_mask: torch.Tensor
    attention_bias: Optional[torch.Tensor]
    salience_logits: Optional[torch.Tensor]
    loss_salience: torch.Tensor
    layer_weights: None = None


def balanced_salience_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    *,
    ranking_weight: float = 0.0,
) -> torch.Tensor:
    """Balanced pointwise supervision plus within-document evidence ranking.

    The pointwise term calibrates evidence probabilities.  The optional
    pairwise term matches how the logits are actually consumed: as relative
    attention scores over source units.  It also avoids the all-zero cold
    start where balanced positive and negative pointwise gradients can nearly
    cancel before the evidence features become discriminative.
    """

    width = min(logits.shape[1], labels.shape[1], valid.shape[1])
    logits, labels, valid = logits[:, :width], labels[:, :width], valid[:, :width]
    supervised = valid & labels.ge(0)
    positive = supervised & labels.gt(0.5)
    negative = supervised & labels.le(0.5)
    terms = []
    if bool(positive.any()):
        terms.append(F.softplus(-logits[positive].float()).mean())
    if bool(negative.any()):
        terms.append(F.softplus(logits[negative].float()).mean())
    pointwise = torch.stack(terms).mean() if terms else logits.float().sum() * 0.0
    if ranking_weight <= 0.0:
        return pointwise

    ranking_terms = []
    for row in range(logits.shape[0]):
        positives = logits[row][positive[row]].float()
        negatives = logits[row][negative[row]].float()
        if positives.numel() and negatives.numel():
            differences = positives[:, None] - negatives[None, :]
            ranking_terms.append(F.softplus(-differences).mean())
    if not ranking_terms:
        return pointwise
    return pointwise + float(ranking_weight) * torch.stack(ranking_terms).mean()


class EvidenceBridge(nn.Module):
    def __init__(self, encoder_hidden: int, decoder_hidden: int, config: dict):
        super().__init__()
        if encoder_hidden == decoder_hidden:
            self.projection: nn.Module = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.RMSNorm(encoder_hidden),
                nn.Linear(encoder_hidden, decoder_hidden, bias=False),
            )
            nn.init.xavier_uniform_(self.projection[-1].weight)
        gate_init = float(config.get("salience_gate_init", 0.1))
        self.salience_attention_gate = nn.Parameter(torch.tensor(math.atanh(gate_init), dtype=torch.float32))
        self.salience_bias_scale = float(config.get("salience_bias_scale", 1.0))
        self.salience_ranking_weight = float(config.get("salience_ranking_weight", 0.0))

    def forward(
        self,
        encoder_memory: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_ids: Optional[torch.Tensor],
        unit_logits: Optional[torch.Tensor],
        valid_units: Optional[torch.Tensor],
        evidence_labels: Optional[torch.Tensor],
    ) -> BridgeOutput:
        memory = self.projection(encoder_memory)
        memory = memory.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
        zero = memory.float().sum() * 0.0
        if unit_ids is None or unit_logits is None or valid_units is None:
            return BridgeOutput(memory, attention_mask, None, unit_logits, zero)
        loss = zero
        if evidence_labels is not None:
            loss = balanced_salience_loss(
                unit_logits,
                evidence_labels,
                valid_units,
                ranking_weight=self.salience_ranking_weight,
            )
        token_bias, source_tokens = unit_evidence_token_bias(
            unit_logits,
            valid_units,
            unit_ids,
            attention_mask,
            scale=self.salience_bias_scale,
        )
        neutral_tokens = attention_mask.bool() & unit_ids.eq(0)
        neutral_count = neutral_tokens.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        neutral_bias = -neutral_count.log().expand_as(token_bias)
        token_bias = torch.where(neutral_tokens, neutral_bias, token_bias)
        gate = torch.tanh(self.salience_attention_gate.float()).to(token_bias.dtype)
        bias = gate * token_bias
        routed_tokens = source_tokens | neutral_tokens
        bias = bias.masked_fill(~routed_tokens, 0).to(memory.dtype)
        return BridgeOutput(memory, attention_mask, bias, unit_logits, loss)

    def bidirectional_gate_mean(self) -> torch.Tensor:
        return self.salience_attention_gate.float().new_zeros(())
