"""Single-memory evidence bridge."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import unit_evidence_token_bias


@dataclass
class BridgeOutput:
    memory: torch.Tensor
    memory_mask: torch.Tensor
    attention_bias: Optional[torch.Tensor]
    salience_logits: Optional[torch.Tensor]
    loss_salience: torch.Tensor
    unit_ids: Optional[torch.Tensor] = None
    valid_units: Optional[torch.Tensor] = None
    layer_weights: None = None
    projection_residual_ratio: Optional[torch.Tensor] = None
    salience_attention_gate: Optional[torch.Tensor] = None
    positive_attention_prior: Optional[torch.Tensor] = None
    negative_attention_prior: Optional[torch.Tensor] = None
    positive_attention_prior_gap: Optional[torch.Tensor] = None


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
        self.trainable_identity_projection = bool(config.get("trainable_identity_projection", False))
        if encoder_hidden == decoder_hidden and not self.trainable_identity_projection:
            self.projection: nn.Module = nn.Identity()
        elif encoder_hidden == decoder_hidden:
            self.projection = IdentityInitializedResidualProjection(encoder_hidden)
        else:
            self.projection = nn.Sequential(
                DtypeSafeRMSNorm(encoder_hidden),
                DtypeSafeLinear(encoder_hidden, decoder_hidden, bias=False),
            )
            nn.init.xavier_uniform_(self.projection[-1].weight)
        gate_init = float(config.get("salience_gate_init", 0.1))
        self.salience_gate_parameterization = str(config.get("salience_gate_parameterization", "sigmoid"))
        if self.salience_gate_parameterization == "sigmoid":
            if not 0.0 < gate_init < 1.0:
                raise ValueError("sigmoid salience_gate_init must be in (0, 1)")
            gate_parameter = math.log(gate_init / (1.0 - gate_init))
        elif self.salience_gate_parameterization == "signed_tanh":
            gate_parameter = math.atanh(gate_init)
        else:
            raise ValueError("bridge.salience_gate_parameterization must be 'signed_tanh' or 'sigmoid'")
        self.salience_attention_gate = nn.Parameter(torch.tensor(gate_parameter, dtype=torch.float32))
        self.salience_bias_scale = float(config.get("salience_bias_scale", 1.0))
        if self.salience_bias_scale <= 0.0:
            raise ValueError("salience_bias_scale must be positive")
        self.salience_length_normalization = str(config.get("salience_length_normalization", "legacy_gated"))
        if self.salience_length_normalization not in {"legacy_gated", "unit_invariant"}:
            raise ValueError("bridge.salience_length_normalization must be 'legacy_gated' or 'unit_invariant'")
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
        projected_memory = self.projection(encoder_memory)
        projection_residual_ratio = self._projection_residual_ratio(projected_memory, encoder_memory)
        memory = projected_memory
        memory = memory.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
        zero = memory.float().sum() * 0.0
        if unit_ids is None or unit_logits is None or valid_units is None:
            return BridgeOutput(
                memory,
                attention_mask,
                None,
                unit_logits,
                zero,
                projection_residual_ratio=projection_residual_ratio.detach(),
                salience_attention_gate=self.attention_gate().detach(),
                positive_attention_prior=zero.detach(),
                negative_attention_prior=zero.detach(),
                positive_attention_prior_gap=zero.detach(),
            )
        loss = zero
        if evidence_labels is not None:
            loss = balanced_salience_loss(
                unit_logits,
                evidence_labels,
                valid_units,
                ranking_weight=self.salience_ranking_weight,
            )
        bias, gate = self._build_attention_bias(
            memory_mask=attention_mask,
            unit_ids=unit_ids,
            unit_logits=unit_logits,
            valid_units=valid_units,
            memory_dtype=memory.dtype,
        )
        return BridgeOutput(
            memory,
            attention_mask,
            bias,
            unit_logits,
            loss,
            unit_ids=unit_ids,
            valid_units=valid_units,
            projection_residual_ratio=projection_residual_ratio.detach(),
            salience_attention_gate=gate.detach(),
            **self._attention_prior_diagnostics(unit_logits, evidence_labels, valid_units, gate),
        )

    def reroute(
        self,
        bridge: BridgeOutput,
        unit_logits: torch.Tensor,
        *,
        evidence_labels: Optional[torch.Tensor] = None,
        loss_salience: Optional[torch.Tensor] = None,
    ) -> BridgeOutput:
        """Replace only the sentence prior of an already projected memory.

        This is the key operation for the dynamic half of DualBridge.  The
        encoder output and PPLX→Qwen projection are reused exactly once; only
        the lightweight unit logits are fused and turned into the SDPA bias.
        """

        if bridge.unit_ids is None or bridge.valid_units is None:
            raise ValueError("Cannot reroute a bridge without unit routing metadata")
        bias, gate = self._build_attention_bias(
            memory_mask=bridge.memory_mask,
            unit_ids=bridge.unit_ids,
            unit_logits=unit_logits,
            valid_units=bridge.valid_units,
            memory_dtype=bridge.memory.dtype,
        )
        return BridgeOutput(
            bridge.memory,
            bridge.memory_mask,
            bias,
            unit_logits,
            bridge.loss_salience if loss_salience is None else loss_salience,
            unit_ids=bridge.unit_ids,
            valid_units=bridge.valid_units,
            projection_residual_ratio=bridge.projection_residual_ratio,
            salience_attention_gate=gate.detach(),
            **self._attention_prior_diagnostics(unit_logits, evidence_labels, bridge.valid_units, gate),
        )

    def _build_attention_bias(
        self,
        *,
        memory_mask: torch.Tensor,
        unit_ids: torch.Tensor,
        unit_logits: torch.Tensor,
        valid_units: torch.Tensor,
        memory_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the one fused token prior consumed by all decoder layers."""
        gate = self.attention_gate()
        if self.salience_length_normalization == "unit_invariant":
            token_bias, source_tokens = unit_evidence_token_bias(
                unit_logits,
                valid_units,
                unit_ids,
                memory_mask,
                scale=self.salience_bias_scale,
                evidence_gate=gate,
            )
        else:
            token_bias, source_tokens = unit_evidence_token_bias(
                unit_logits,
                valid_units,
                unit_ids,
                memory_mask,
                scale=self.salience_bias_scale,
            )
        neutral_tokens = memory_mask.bool() & unit_ids.eq(0)
        neutral_count = neutral_tokens.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        neutral_bias = -neutral_count.log().expand_as(token_bias)
        if self.salience_length_normalization == "legacy_gated":
            token_bias = gate.to(token_bias.dtype) * torch.where(neutral_tokens, neutral_bias, token_bias)
        else:
            token_bias = torch.where(neutral_tokens, neutral_bias, token_bias)
        bias = token_bias
        routed_tokens = source_tokens | neutral_tokens
        bias = bias.masked_fill(~routed_tokens, 0).to(memory_dtype)
        return bias, gate

    def bidirectional_gate_mean(self) -> torch.Tensor:
        return self.salience_attention_gate.float().new_zeros(())

    def attention_gate(self) -> torch.Tensor:
        """Return the scalar strength applied to evidence unit logits."""

        parameter = self.salience_attention_gate.float()
        if self.salience_gate_parameterization == "sigmoid":
            return torch.sigmoid(parameter)
        return torch.tanh(parameter)

    def unit_attention_energy(self, unit_logits: torch.Tensor) -> torch.Tensor:
        """Return the live, relative unit energy consumed by the bridge.

        For the corrected unit-invariant bridge, every token in a unit also
        receives a ``-log(token_count)`` term.  That term deliberately
        cancels when the token scores are summed into the unit's total
        softmax mass.  The remaining relative energy is precisely this
        gate/scale/clipped-logit value.  Keeping it as a live tensor (rather
        than reusing a detached diagnostic) lets an attention-aligned
        contrastive loss update the same gate and clip-bounded quantity.
        """

        return self.attention_gate().float() * float(self.salience_bias_scale) * unit_logits.float().clamp(-5.0, 5.0)

    def _attention_prior_diagnostics(
        self,
        unit_logits: torch.Tensor,
        evidence_labels: Optional[torch.Tensor],
        valid_units: torch.Tensor,
        gate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Log the unit-level additive prior consumed by cross-attention.

        This is deliberately a *prior* diagnostic, not a claim that every
        decoder query gives all probability mass to evidence.  Content-based
        cross-attention is still free to retrieve any source token.  It does,
        however, directly verifies the supervised path used to favour
        predicted positives at every decoder layer.
        """

        zero = unit_logits.float().sum() * 0.0
        result = {
            "positive_attention_prior": zero.detach(),
            "negative_attention_prior": zero.detach(),
            "positive_attention_prior_gap": zero.detach(),
        }
        if evidence_labels is None:
            return result
        width = min(unit_logits.shape[1], evidence_labels.shape[1], valid_units.shape[1])
        if width <= 0:
            return result
        logits = unit_logits[:, :width].float().clamp(-5.0, 5.0)
        labels = evidence_labels[:, :width]
        valid = valid_units[:, :width].bool() & labels.ge(0.0)
        positive = valid & labels.gt(0.5)
        negative = valid & labels.le(0.5)
        prior = self.unit_attention_energy(logits).detach()
        positive_count = positive.sum()
        negative_count = negative.sum()
        has_both = positive_count.gt(0) & negative_count.gt(0)
        pos = (prior * positive.to(prior.dtype)).sum() / positive_count.to(prior.dtype).clamp_min(1.0)
        neg = (prior * negative.to(prior.dtype)).sum() / negative_count.to(prior.dtype).clamp_min(1.0)
        pos = torch.where(has_both, pos, zero.detach())
        neg = torch.where(has_both, neg, zero.detach())
        return {
            "positive_attention_prior": pos,
            "negative_attention_prior": neg,
            "positive_attention_prior_gap": (pos - neg).detach(),
        }

    @staticmethod
    def _projection_residual_ratio(
        projected_memory: torch.Tensor,
        encoder_memory: torch.Tensor,
        *,
        max_token_samples: int = 64,
    ) -> torch.Tensor:
        """Cheap, detached coordinate-correction diagnostic.

        A full FP32 ``[B,S,D]`` difference would add gigabytes of temporary
        memory on PubMed.  A deterministic, evenly-spaced token sample is
        sufficient to reveal whether the identity-initialized correction is
        still near zero, without changing the training graph or peak VRAM.
        """

        if projected_memory.shape[-1] != encoder_memory.shape[-1]:
            return projected_memory.new_zeros((), dtype=torch.float32)
        sequence_length = int(projected_memory.shape[1])
        stride = max(1, (sequence_length + int(max_token_samples) - 1) // int(max_token_samples))
        with torch.no_grad():
            projected_sample = projected_memory.detach()[:, ::stride].float()
            encoder_sample = encoder_memory.detach()[:, ::stride].float()
            residual_rms = (projected_sample - encoder_sample).square().mean().sqrt()
            reference_rms = encoder_sample.square().mean().sqrt().clamp_min(1.0e-8)
            return residual_rms / reference_rms


class IdentityInitializedResidualProjection(nn.Module):
    """Learn an encoder-to-decoder coordinate correction without a cold start.

    ``update`` is intentionally zero-initialized.  Therefore this module is
    exactly the identity before its first optimizer step, unlike a Xavier
    projection.  Unlike a scalar-gated residual initialized at zero, the
    update matrix itself receives gradients immediately.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.update = ZeroInitializedLinear(hidden_size, hidden_size)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        normalized = F.rms_norm(
            memory,
            self.norm.normalized_shape,
            self.norm.weight.to(dtype=memory.dtype),
            self.norm.eps,
        )
        return memory + self.update(normalized)


class ZeroInitializedLinear(nn.Module):
    """Bias-free linear map whose construction is RNG-neutral."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.weight.to(dtype=values.dtype))


class DtypeSafeRMSNorm(nn.RMSNorm):
    """RMSNorm with FP32 master weights safe for standalone BF16/FP16 use."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            values,
            self.normalized_shape,
            self.weight.to(dtype=values.dtype),
            self.eps,
        )


class DtypeSafeLinear(nn.Linear):
    """Linear layer that casts master weights at the operation boundary."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(dtype=values.dtype) if self.bias is not None else None
        return F.linear(values, self.weight.to(dtype=values.dtype), bias)
