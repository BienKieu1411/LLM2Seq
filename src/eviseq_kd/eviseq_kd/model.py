"""Independent EviSeq-KD model wrapper around its bundled student graph."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .kd import top1_agreement, topk_distillation_loss
from .student.modeling.architecture import EviSeq


class EviSeqKD(nn.Module):
    """Add offline sequence KD and optional top-k logit KD around the bundled EviSeq graph.

    Gold batches use the complete original EviSeq objective, including its
    evidence-focused contrastive loss.  The pseudo branch uses CE only, so
    gold evidence labels and pseudo targets cannot contaminate one another.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.base = EviSeq(config)
        distillation = config.get("training", {}).get("distillation", {})
        self.distillation_enabled = bool(distillation.get("enabled", False))
        self.sequence_enabled = bool(distillation.get("sequence_enabled", True))
        self.sequence_weight = float(distillation.get("sequence_weight", 0.0)) if self.sequence_enabled else 0.0
        self.logit_enabled = bool(distillation.get("logit_enabled", False))
        self.logit_weight = float(distillation.get("logit_weight", 0.0)) if self.logit_enabled else 0.0
        self.kd_temperature = float(distillation.get("temperature", 2.0))
        if self.logit_enabled and not self.sequence_enabled:
            raise ValueError("Logit KD requires sequence_enabled=true so its positions have a pseudo sequence")
        if self.distillation_enabled and self.sequence_enabled and self.sequence_weight <= 0.0:
            raise ValueError("Enabled sequence KD requires training.distillation.sequence_weight > 0")
        if self.distillation_enabled and self.logit_enabled and self.logit_weight <= 0.0:
            raise ValueError("Enabled logit KD requires training.distillation.logit_weight > 0")
        if self.kd_temperature <= 0.0:
            raise ValueError("training.distillation.temperature must be positive")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        unit_ids: Optional[torch.Tensor] = None,
        evidence_labels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pseudo_decoder_input_ids: Optional[torch.Tensor] = None,
        pseudo_decoder_attention_mask: Optional[torch.Tensor] = None,
        pseudo_labels: Optional[torch.Tensor] = None,
        teacher_topk_ids: Optional[torch.Tensor] = None,
        teacher_topk_logits: Optional[torch.Tensor] = None,
        teacher_kd_mask: Optional[torch.Tensor] = None,
        compute_source_diagnostics: bool = False,
        contrastive_mode: str = "local",
    ) -> Dict[str, torch.Tensor]:
        pseudo_present = pseudo_decoder_input_ids is not None or pseudo_labels is not None
        gold = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            unit_ids=unit_ids,
            evidence_labels=evidence_labels,
            labels=labels,
            compute_source_diagnostics=compute_source_diagnostics,
            contrastive_mode=contrastive_mode,
        )
        if contrastive_mode == "representations_only" and pseudo_present:
            # The legacy GradCache pass asks only for representations.  Its
            # extra batch fields must not trigger a second pseudo forward.
            return gold
        if labels is None:
            return gold
        zero = gold["loss"].float() * 0.0
        loss_pseudo = zero
        weighted_pseudo = zero
        loss_kd = zero
        weighted_kd = zero
        kd_agreement = zero.detach()

        if pseudo_present:
            if pseudo_decoder_input_ids is None or pseudo_labels is None:
                raise ValueError("pseudo_decoder_input_ids and pseudo_labels must be provided together")
            if not self.distillation_enabled or not self.sequence_enabled:
                raise ValueError("A pseudo branch was supplied but sequence KD is disabled")
            if pseudo_decoder_attention_mask is None:
                pseudo_decoder_attention_mask = pseudo_decoder_input_ids.ne(0)

            # Disable only the document-level contrastive contribution in this
            # auxiliary branch. Evidence contrastive is already disabled by
            # evidence_labels=None; the gold branch remains unchanged.
            previous_scale = self.base._contrastive_scale
            self.base.set_contrastive_scale(0.0)
            try:
                pseudo = self.base(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=pseudo_decoder_input_ids,
                    decoder_attention_mask=pseudo_decoder_attention_mask,
                    unit_ids=unit_ids,
                    evidence_labels=None,
                    labels=pseudo_labels,
                    contrastive_mode="local",
                )
            finally:
                self.base.set_contrastive_scale(previous_scale)
            loss_pseudo = pseudo["loss_ce"]
            weighted_pseudo = float(self.sequence_weight) * loss_pseudo

            topk_present = (
                teacher_topk_ids is not None or teacher_topk_logits is not None or teacher_kd_mask is not None
            )
            if topk_present:
                if not self.distillation_enabled or not self.logit_enabled:
                    raise ValueError("Teacher top-k tensors were supplied but logit KD is disabled")
                if teacher_topk_ids is None or teacher_topk_logits is None:
                    raise ValueError("teacher_topk_ids and teacher_topk_logits must be supplied together")
                logits = self.base(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=pseudo_decoder_input_ids,
                    decoder_attention_mask=pseudo_decoder_attention_mask,
                    unit_ids=unit_ids,
                    evidence_labels=None,
                    labels=None,
                )["logits"]
                loss_kd = topk_distillation_loss(
                    logits,
                    teacher_topk_ids,
                    teacher_topk_logits,
                    mask=teacher_kd_mask,
                    temperature=self.kd_temperature,
                )
                kd_agreement = top1_agreement(logits, teacher_topk_ids, mask=teacher_kd_mask).detach()
                weighted_kd = float(self.logit_weight) * loss_kd

        result = dict(gold)
        result["loss_pseudo"] = loss_pseudo.detach()
        result["weighted_pseudo"] = weighted_pseudo.detach()
        result["loss_kd"] = loss_kd.detach()
        result["weighted_kd"] = weighted_kd.detach()
        result["kd_top1_agreement"] = kd_agreement
        result["loss"] = gold["loss"] + weighted_pseudo + weighted_kd
        return result

    def set_training_stage(self, stage: str) -> None:
        self.base.set_training_stage(stage)

    def __getattr__(self, name: str) -> Any:
        """Expose legacy trainer/model diagnostics through the composition wrapper."""

        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = object.__getattribute__(self, "_modules")
            base = modules.get("base")
            if base is not None and hasattr(base, name):
                return getattr(base, name)
            raise

    def set_contrastive_scale(self, scale: float) -> None:
        self.base.set_contrastive_scale(scale)

    def set_evidence_contrastive_scale(self, scale: float) -> None:
        self.base.set_evidence_contrastive_scale(scale)

    def parameter_summary(self) -> Dict[str, int]:
        return self.base.parameter_summary()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
        """Accept both a legacy EviSeq state dict and an EviSeq-KD state dict."""

        expected = set(self.state_dict())
        if not any(key in expected for key in state_dict) and any(not key.startswith("base.") for key in state_dict):
            state_dict = {key if key.startswith("base.") else f"base.{key}": value for key, value in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
