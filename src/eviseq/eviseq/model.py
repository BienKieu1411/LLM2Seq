"""One encoder, one evidence interface, and one pretrained causal decoder."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bridge import BridgeOutput, EvidenceBridge
from .contrastive import (
    SourcePromptAlignmentHead,
    exact_duplicate_mask,
    info_nce_loss,
    last_prompt_states,
)
from .decoder import PretrainedQwenDecoder
from .encoder import build_encoder


def torch_dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in values:
        raise ValueError(f"Unsupported dtype {name!r}")
    return values[name]


class EviSeq(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_config = config["model"]
        dtype = torch_dtype(str(model_config.get("dtype", "float32")))
        checkpointing = bool(model_config.get("gradient_checkpointing", True))
        self.encoder = build_encoder(config, dtype)
        expected_encoder = int(model_config["encoder_hidden_size"])
        if self.encoder.hidden_size != expected_encoder:
            raise ValueError(
                f"Configured encoder hidden size {expected_encoder} != checkpoint {self.encoder.hidden_size}"
            )
        decoder_hidden = int(model_config["decoder_hidden_size"])
        self.adapter = EvidenceBridge(expected_encoder, decoder_hidden, config["bridge"])
        self.decoder = PretrainedQwenDecoder(str(model_config["decoder_name"]), config["decoder"], dtype, checkpointing)
        if int(self.decoder.config.hidden_size) != decoder_hidden:
            raise ValueError("Configured decoder hidden size does not match checkpoint")
        objectives = config.get("objectives", {})
        self.salience_weight = float(objectives.get("salience_weight", 0.1))
        self.use_contrastive = bool(objectives.get("use_contrastive", True))
        self.contrastive_weight = float(objectives.get("contrastive_weight", 0.05)) if self.use_contrastive else 0.0
        self.contrastive_temperature = float(objectives.get("contrastive_temperature", 0.07))
        self.contrastive_warmup_epochs = int(objectives.get("contrastive_warmup_epochs", 2))
        self.contrastive_across_accumulation = bool(objectives.get("contrastive_across_accumulation", True))
        if self.use_contrastive:
            self.alignment_head: Optional[SourcePromptAlignmentHead] = SourcePromptAlignmentHead(
                decoder_hidden,
                projection_size=int(objectives.get("contrastive_projection_size", 256)),
                pooling=str(objectives.get("contrastive_pooling", "mean_last")),
            )
        else:
            self.alignment_head = None
        self._contrastive_scale = 1.0
        self._stage = "unconfigured"

    @property
    def lm_head(self) -> nn.Module:
        return self.decoder.lm_head

    def set_contrastive_scale(self, scale: float) -> None:
        self._contrastive_scale = max(0.0, min(1.0, float(scale)))

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
        evidence_labels: Optional[torch.Tensor] = None,
    ) -> BridgeOutput:
        encoded = self.encoder(input_ids, attention_mask, unit_ids=unit_ids)
        return self.adapter(
            encoded.memory,
            attention_mask,
            unit_ids,
            encoded.unit_logits,
            encoded.valid_units,
            evidence_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        unit_ids: Optional[torch.Tensor] = None,
        evidence_labels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compute_source_diagnostics: bool = False,
        contrastive_mode: str = "local",
    ) -> Dict[str, torch.Tensor]:
        if contrastive_mode not in {"local", "representations_only", "deferred"}:
            raise ValueError(f"Unknown contrastive mode: {contrastive_mode}")
        encoded = self.encoder(input_ids, attention_mask, unit_ids=unit_ids)
        bridge = self.adapter(
            encoded.memory,
            attention_mask,
            unit_ids,
            encoded.unit_logits,
            encoded.valid_units,
            evidence_labels,
        )
        states, _ = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=bridge.memory,
            encoder_attention_mask=bridge.memory_mask,
            encoder_attention_bias=bridge.attention_bias,
            use_cache=False,
        )
        if labels is None:
            if contrastive_mode != "local":
                raise ValueError("Cached contrastive modes require decoder labels to locate the fixed prompt state")
            return {"logits": self.lm_head(states)}
        supervised = labels.ne(-100)
        if not bool(supervised.any()):
            raise ValueError("A training batch must contain supervised decoder labels")
        representations: Optional[Dict[str, torch.Tensor]] = None
        measure_source_use = self.alignment_head is not None and (
            self.training or bool(compute_source_diagnostics) or contrastive_mode != "local"
        )
        if measure_source_use:
            assert self.alignment_head is not None
            representations = self.alignment_head(
                bridge.memory,
                bridge.memory_mask,
                last_prompt_states(states, labels),
            )
        if contrastive_mode == "representations_only":
            if representations is None:
                raise RuntimeError("representations_only requires enabled contrastive learning")
            return representations
        logits = self.lm_head(states[supervised])
        loss_ce = F.cross_entropy(logits.float(), labels[supervised])
        loss_contrastive = states.float().sum() * 0.0
        prompt_retrieval_accuracy = loss_contrastive.detach()
        if representations is not None and contrastive_mode == "local":
            loss_contrastive, prompt_retrieval_accuracy = info_nce_loss(
                representations["source_repr"],
                representations["prompt_repr"],
                self.contrastive_temperature,
                duplicate_mask=exact_duplicate_mask(input_ids, attention_mask),
            )
        auxiliary_scale = 1.0 if self.training else 0.0
        weighted_salience = self.salience_weight * bridge.loss_salience.float()
        local_contrastive_scale = self._contrastive_scale if contrastive_mode == "local" else 0.0
        weighted_contrastive = self.contrastive_weight * local_contrastive_scale * loss_contrastive.float()
        loss = loss_ce + auxiliary_scale * (weighted_salience + weighted_contrastive)
        result: Dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_salience": bridge.loss_salience,
            "weighted_salience": weighted_salience.detach(),
            "salience_to_ce_ratio": (weighted_salience.detach() / loss_ce.detach().float().clamp_min(1.0e-8)),
            "loss_contrastive": loss_contrastive,
            "weighted_contrastive": weighted_contrastive.detach(),
            "contrastive_to_ce_ratio": (weighted_contrastive.detach() / loss_ce.detach().float().clamp_min(1.0e-8)),
            "prompt_retrieval_accuracy": prompt_retrieval_accuracy,
            "contrastive_candidates": states.new_tensor(input_ids.shape[0] if self.alignment_head is not None else 0),
            "contrastive_scale": states.new_tensor(self._contrastive_scale),
            "cross_gate_mean": self.decoder.cross_gate_mean().detach(),
            "cross_residual_ratio": self.decoder.cross_residual_ratio_mean().detach(),
            "bidirectional_gate_mean": encoded.native_gate_mean.detach(),
            "projection_gate": encoded.native_gate_mean.detach(),
            "evidence_view_gate": encoded.native_gate_mean.detach(),
        }
        if contrastive_mode == "deferred":
            if representations is None:
                raise RuntimeError("deferred contrastive backward requires enabled contrastive learning")
            result.update(representations)
        if bridge.salience_logits is not None and evidence_labels is not None:
            width = min(bridge.salience_logits.shape[1], evidence_labels.shape[1])
            valid = evidence_labels[:, :width].ge(0)
            if bool(valid.any()):
                probabilities = torch.sigmoid(bridge.salience_logits[:, :width][valid].float())
                predictions = probabilities.ge(0.5)
                gold = evidence_labels[:, :width][valid].gt(0.5)
                tp = (predictions & gold).sum().float()
                result["salience_probability_mean"] = probabilities.mean().detach()
                result["salience_predicted_positive_rate"] = predictions.float().mean().detach()
                result["salience_tp"] = tp.detach()
                result["salience_predicted_count"] = predictions.sum().float().detach()
                result["salience_gold_count"] = gold.sum().float().detach()
                result["salience_precision"] = (tp / predictions.sum().float().clamp_min(1)).detach()
                result["salience_recall"] = (tp / gold.sum().float().clamp_min(1)).detach()
                correct_pairs = probabilities.new_zeros(())
                pair_count = probabilities.new_zeros(())
                logits = bridge.salience_logits[:, :width].float()
                for row in range(logits.shape[0]):
                    row_valid = valid[row]
                    row_gold = evidence_labels[row, :width].gt(0.5) & row_valid
                    row_negative = ~evidence_labels[row, :width].gt(0.5) & row_valid
                    positives = logits[row][row_gold]
                    negatives = logits[row][row_negative]
                    if positives.numel() and negatives.numel():
                        differences = positives[:, None] - negatives[None, :]
                        correct_pairs = correct_pairs + differences.gt(0).float().sum()
                        correct_pairs = correct_pairs + 0.5 * differences.eq(0).float().sum()
                        pair_count = pair_count + differences.new_tensor(differences.numel())
                result["salience_correct_pairs"] = correct_pairs.detach()
                result["salience_pair_count"] = pair_count.detach()
        if self.alignment_head is not None and self.alignment_head.pool_gate is not None:
            result["alignment_last_pool_weight"] = torch.sigmoid(self.alignment_head.pool_gate.float()).mean().detach()
        return result

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"interface_warmup", "full_finetune"}:
            raise ValueError(f"Unknown training stage: {stage}")
        full = stage == "full_finetune"
        self.encoder.set_trainable(full)
        self.decoder.set_backbone_trainable(full)
        for parameter in self.adapter.parameters():
            parameter.requires_grad = True
        if self.alignment_head is not None:
            for parameter in self.alignment_head.parameters():
                parameter.requires_grad = True
        self._stage = stage
        if full:
            frozen = [name for name, value in self.named_parameters() if not value.requires_grad]
            if frozen:
                raise RuntimeError(f"Full fine-tuning left parameters frozen: {frozen[:20]}")

    def parameter_summary(self) -> Dict[str, int]:
        return {
            "encoder": sum(value.numel() for value in self.encoder.parameters()),
            "adapter": sum(value.numel() for value in self.adapter.parameters()),
            "contrastive_head": (
                sum(value.numel() for value in self.alignment_head.parameters())
                if self.alignment_head is not None
                else 0
            ),
            "decoder": sum(value.numel() for value in self.decoder.parameters()),
            "total": sum(value.numel() for value in self.parameters()),
            "trainable": sum(value.numel() for value in self.parameters() if value.requires_grad),
        }
