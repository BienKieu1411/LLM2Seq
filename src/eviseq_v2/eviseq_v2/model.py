"""One encoder, one evidence interface, and one pretrained causal decoder.

EviSeq V2 adds:
- Evidence-focused hard contrastive learning (replaces document-level InfoNCE)
- BRIO-like candidate ranking via teacher-forced log probabilities
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bridge import BridgeOutput, EvidenceBridge
from .contrastive import (
    EvidenceContrastiveHead,
    SourcePromptAlignmentHead,
    decoder_summary_representation,
    evidence_info_nce_loss,
    exact_duplicate_mask,
    info_nce_loss,
    last_prompt_states,
    length_normalized_log_prob,
    pairwise_ranking_loss,
)
from .decoder import PretrainedQwenDecoder
from .encoder import build_encoder
from .native_attention import pool_units


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

        # --- Salience ---
        self.salience_weight = float(objectives.get("salience_weight", 0.1))

        # --- Document-level contrastive (V1, disabled by default in V2) ---
        self.use_contrastive = bool(objectives.get("use_contrastive", False))
        self.contrastive_weight = float(objectives.get("contrastive_weight", 0.0)) if self.use_contrastive else 0.0
        self.contrastive_temperature = float(objectives.get("contrastive_temperature", 0.07))
        self.contrastive_warmup_epochs = int(objectives.get("contrastive_warmup_epochs", 2))
        self.contrastive_across_accumulation = bool(objectives.get("contrastive_across_accumulation", False))
        if self.use_contrastive:
            self.alignment_head: Optional[SourcePromptAlignmentHead] = SourcePromptAlignmentHead(
                decoder_hidden,
                projection_size=int(objectives.get("contrastive_projection_size", 256)),
                pooling=str(objectives.get("contrastive_pooling", "mean_last")),
            )
        else:
            self.alignment_head = None

        # --- Evidence-focused contrastive (V2 new) ---
        self.use_evidence_contrastive = bool(objectives.get("use_evidence_contrastive", True))
        self.evidence_contrastive_weight = float(objectives.get("evidence_contrastive_weight", 0.05))
        self.evidence_contrastive_temperature = float(objectives.get("evidence_contrastive_temperature", 0.07))
        self.evidence_hard_negatives = int(objectives.get("evidence_hard_negatives", 4))
        self.evidence_contrastive_warmup_epochs = int(objectives.get("evidence_contrastive_warmup_epochs", 2))
        if self.use_evidence_contrastive:
            self.evidence_contrastive_head: Optional[EvidenceContrastiveHead] = EvidenceContrastiveHead(
                encoder_hidden_size=expected_encoder,
                decoder_hidden_size=decoder_hidden,
                projection_size=int(objectives.get("evidence_contrastive_projection_size", 256)),
            )
        else:
            self.evidence_contrastive_head = None

        # --- Candidate ranking (V2 new, Phase 3) ---
        ranking_config = config.get("ranking", {})
        self.ranking_enabled = bool(ranking_config.get("enabled", False))
        self.ranking_weight = float(ranking_config.get("weight", 0.10))
        self.ranking_margin = float(ranking_config.get("margin", 0.01))
        self.ranking_length_penalty_alpha = float(ranking_config.get("length_penalty_alpha", 1.0))
        self.ranking_minimum_quality_gap = float(ranking_config.get("minimum_quality_gap", 0.5))

        self._contrastive_scale = 1.0
        self._evidence_contrastive_scale = 1.0
        self._stage = "unconfigured"

    @property
    def lm_head(self) -> nn.Module:
        return self.decoder.lm_head

    def set_contrastive_scale(self, scale: float) -> None:
        self._contrastive_scale = max(0.0, min(1.0, float(scale)))

    def set_evidence_contrastive_scale(self, scale: float) -> None:
        self._evidence_contrastive_scale = max(0.0, min(1.0, float(scale)))

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

        # --- Document-level contrastive (V1 path, if enabled) ---
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

        # --- CE loss ---
        logits = self.lm_head(states[supervised])
        loss_ce = F.cross_entropy(logits.float(), labels[supervised])

        # --- Document-level contrastive loss (V1) ---
        loss_contrastive = states.float().sum() * 0.0
        prompt_retrieval_accuracy = loss_contrastive.detach()
        if representations is not None and contrastive_mode == "local":
            loss_contrastive, prompt_retrieval_accuracy = info_nce_loss(
                representations["source_repr"],
                representations["prompt_repr"],
                self.contrastive_temperature,
                duplicate_mask=exact_duplicate_mask(input_ids, attention_mask),
            )

        # --- Evidence-focused contrastive (V2 new) ---
        evi_results: Dict[str, torch.Tensor] = {}
        loss_evidence_contrastive = states.float().sum() * 0.0
        if (
            self.evidence_contrastive_head is not None
            and self.use_evidence_contrastive
            and self.training
            and unit_ids is not None
            and evidence_labels is not None
            and encoded.unit_logits is not None
            and encoded.valid_units is not None
        ):
            # Get sentence-level representations from encoder
            unit_count = int(unit_ids.max().item()) if unit_ids is not None else 0
            if unit_count > 0:
                sentence_reprs, valid_units = pool_units(encoded.memory, unit_ids, unit_count)

                # Get summary representation from decoder
                summary_repr = decoder_summary_representation(states, labels)

                # Project through evidence contrastive head
                q, k = self.evidence_contrastive_head(summary_repr, sentence_reprs)

                # Within-document evidence InfoNCE
                evi_results = evidence_info_nce_loss(
                    query=q,
                    keys=k,
                    evidence_labels=evidence_labels,
                    valid_units=valid_units,
                    temperature=self.evidence_contrastive_temperature,
                    num_hard_negatives=self.evidence_hard_negatives,
                    salience_logits=encoded.unit_logits,
                )
                loss_evidence_contrastive = evi_results.get("evidence_contrastive_loss", loss_evidence_contrastive)

        # --- Total loss ---
        auxiliary_scale = 1.0 if self.training else 0.0
        weighted_salience = self.salience_weight * bridge.loss_salience.float()

        local_contrastive_scale = self._contrastive_scale if contrastive_mode == "local" else 0.0
        weighted_contrastive = self.contrastive_weight * local_contrastive_scale * loss_contrastive.float()

        evidence_scale = self._evidence_contrastive_scale
        weighted_evidence = self.evidence_contrastive_weight * evidence_scale * loss_evidence_contrastive.float()

        loss = loss_ce + auxiliary_scale * (weighted_salience + weighted_contrastive + weighted_evidence)

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
            # Evidence contrastive metrics
            "loss_evidence_contrastive": loss_evidence_contrastive.detach(),
            "weighted_evidence_contrastive": weighted_evidence.detach(),
            "evidence_contrastive_scale": states.new_tensor(evidence_scale),
            "cross_gate_mean": self.decoder.cross_gate_mean().detach(),
            "cross_residual_ratio": self.decoder.cross_residual_ratio_mean().detach(),
            "bidirectional_gate_mean": encoded.native_gate_mean.detach(),
            "projection_gate": encoded.native_gate_mean.detach(),
            "evidence_view_gate": encoded.native_gate_mean.detach(),
        }

        # Evidence contrastive diagnostics
        for key in (
            "evidence_top1_accuracy",
            "positive_similarity",
            "hard_negative_similarity",
            "evidence_similarity_gap",
            "evidence_valid_examples",
        ):
            if key in evi_results:
                result[key] = evi_results[key]
            else:
                result[key] = states.new_tensor(0.0)

        if contrastive_mode == "deferred":
            if representations is None:
                raise RuntimeError("deferred contrastive backward requires enabled contrastive learning")
            result.update(representations)

        # --- Salience diagnostics ---
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
                sal_logits = bridge.salience_logits[:, :width].float()
                for row in range(sal_logits.shape[0]):
                    row_valid = valid[row]
                    row_gold = evidence_labels[row, :width].gt(0.5) & row_valid
                    row_negative = ~evidence_labels[row, :width].gt(0.5) & row_valid
                    positives = sal_logits[row][row_gold]
                    negatives = sal_logits[row][row_negative]
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

    def forward_ranking(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_ids: Optional[torch.Tensor],
        evidence_labels: Optional[torch.Tensor],
        reference_decoder_input_ids: torch.Tensor,
        reference_decoder_attention_mask: Optional[torch.Tensor],
        reference_labels: torch.Tensor,
        candidate_decoder_input_ids: torch.Tensor,
        candidate_decoder_attention_mask: torch.Tensor,
        candidate_labels: torch.Tensor,
        candidate_quality_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for Phase 3: CE on reference + pairwise ranking on candidates.

        Args:
            input_ids: [B, S] source tokens
            attention_mask: [B, S]
            unit_ids: [B, S]
            evidence_labels: [B, U]
            reference_decoder_input_ids: [B, T_ref] teacher-forcing input for reference
            reference_decoder_attention_mask: [B, T_ref]
            reference_labels: [B, T_ref]
            candidate_decoder_input_ids: [B, N, T_cand] for N candidates
            candidate_decoder_attention_mask: [B, N, T_cand]
            candidate_labels: [B, N, T_cand]
            candidate_quality_scores: [B, N] external quality scores
        """
        # Encode source once
        encoded = self.encoder(input_ids, attention_mask, unit_ids=unit_ids)
        bridge = self.adapter(
            encoded.memory,
            attention_mask,
            unit_ids,
            encoded.unit_logits,
            encoded.valid_units,
            evidence_labels,
        )

        # CE on reference
        ref_states, _ = self.decoder(
            input_ids=reference_decoder_input_ids,
            attention_mask=reference_decoder_attention_mask,
            encoder_hidden_states=bridge.memory,
            encoder_attention_mask=bridge.memory_mask,
            encoder_attention_bias=bridge.attention_bias,
            use_cache=False,
        )
        ref_supervised = reference_labels.ne(-100)
        ref_logits = self.lm_head(ref_states[ref_supervised])
        loss_ce = F.cross_entropy(ref_logits.float(), reference_labels[ref_supervised])

        # Score candidates
        batch_size, num_candidates, cand_len = candidate_decoder_input_ids.shape
        all_scores: List[torch.Tensor] = []
        all_quality: List[torch.Tensor] = []

        for b in range(batch_size):
            scores_b: List[torch.Tensor] = []
            quality_b: List[torch.Tensor] = []

            for c in range(num_candidates):
                c_input = candidate_decoder_input_ids[b, c].unsqueeze(0)  # [1, T]
                c_mask = candidate_decoder_attention_mask[b, c].unsqueeze(0)  # [1, T]
                c_labels = candidate_labels[b, c]  # [T]

                if not bool(c_labels.ne(-100).any()):
                    continue

                c_states, _ = self.decoder(
                    input_ids=c_input,
                    attention_mask=c_mask,
                    encoder_hidden_states=bridge.memory[b : b + 1],
                    encoder_attention_mask=bridge.memory_mask[b : b + 1],
                    encoder_attention_bias=bridge.attention_bias[b : b + 1]
                    if bridge.attention_bias is not None
                    else None,
                    use_cache=False,
                )
                c_logits = self.lm_head(c_states.squeeze(0))  # [T, V]
                score = length_normalized_log_prob(
                    c_logits,
                    c_labels,
                    alpha=self.ranking_length_penalty_alpha,
                )
                scores_b.append(score)
                quality_b.append(candidate_quality_scores[b, c])

            if len(scores_b) >= 2:
                all_scores.append(torch.stack(scores_b))
                all_quality.append(torch.stack(quality_b))

        # Pairwise ranking loss
        loss_rank = loss_ce.new_tensor(0.0)
        pair_accuracy = loss_ce.new_tensor(0.0)
        if all_scores:
            rank_losses = []
            accuracies = []
            for scores, quality in zip(all_scores, all_quality):
                rl, ra = pairwise_ranking_loss(
                    scores,
                    quality,
                    margin=self.ranking_margin,
                    minimum_quality_gap=self.ranking_minimum_quality_gap,
                )
                rank_losses.append(rl)
                accuracies.append(ra)
            loss_rank = torch.stack(rank_losses).mean()
            pair_accuracy = torch.stack(accuracies).mean()

        weighted_rank = self.ranking_weight * loss_rank
        loss = loss_ce + weighted_rank

        # Salience loss contribution (lower weight in phase 3)
        weighted_salience = self.salience_weight * bridge.loss_salience.float()
        loss = loss + weighted_salience

        return {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_rank": loss_rank,
            "weighted_rank": weighted_rank.detach(),
            "loss_salience": bridge.loss_salience,
            "weighted_salience": weighted_salience.detach(),
            "candidate_pair_accuracy": pair_accuracy,
            "ranking_to_ce_ratio": (weighted_rank.detach() / loss_ce.detach().float().clamp_min(1e-8)),
            "cross_gate_mean": self.decoder.cross_gate_mean().detach(),
            "cross_residual_ratio": self.decoder.cross_residual_ratio_mean().detach(),
            "bidirectional_gate_mean": encoded.native_gate_mean.detach(),
        }

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"interface_warmup", "full_finetune", "ranking_finetune"}:
            raise ValueError(f"Unknown training stage: {stage}")

        if stage == "ranking_finetune":
            # Phase 3: everything trainable but with lower LRs
            self.encoder.set_trainable(True)
            self.decoder.set_backbone_trainable(True)
            for parameter in self.adapter.parameters():
                parameter.requires_grad = True
            # Evidence contrastive head is NOT needed in ranking phase
            if self.evidence_contrastive_head is not None:
                for parameter in self.evidence_contrastive_head.parameters():
                    parameter.requires_grad = False
            if self.alignment_head is not None:
                for parameter in self.alignment_head.parameters():
                    parameter.requires_grad = False
        else:
            full = stage == "full_finetune"
            self.encoder.set_trainable(full)
            self.decoder.set_backbone_trainable(full)
            for parameter in self.adapter.parameters():
                parameter.requires_grad = True
            if self.alignment_head is not None:
                for parameter in self.alignment_head.parameters():
                    parameter.requires_grad = True
            if self.evidence_contrastive_head is not None:
                for parameter in self.evidence_contrastive_head.parameters():
                    parameter.requires_grad = True

        self._stage = stage

        if stage == "full_finetune":
            # Verify everything except training-only heads is trainable
            training_only_prefixes = ("alignment_head.", "evidence_contrastive_head.")
            frozen = [
                name
                for name, value in self.named_parameters()
                if not value.requires_grad and not any(name.startswith(p) for p in training_only_prefixes)
            ]
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
            "evidence_contrastive_head": (
                sum(value.numel() for value in self.evidence_contrastive_head.parameters())
                if self.evidence_contrastive_head is not None
                else 0
            ),
            "decoder": sum(value.numel() for value in self.decoder.parameters()),
            "total": sum(value.numel() for value in self.parameters()),
            "trainable": sum(value.numel() for value in self.parameters() if value.requires_grad),
        }
