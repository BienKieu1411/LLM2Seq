"""LLM2Seq-v3: encoder-adapter-decoder with source-alignment contrastive learning."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import AdapterOutput, SummaryAdapterV2
from .contrastive import (
    SourceAlignmentHead,
    hard_negative_indices,
    info_nce_loss,
    last_prompt_states,
    per_example_nll,
    source_memory_for_mining,
    source_swap_contrastive_loss,
)
from .decoder import PretrainedQwenDecoder
from .encoder import QwenEmbeddingEncoder


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}")
    return mapping[name]


class LLM2SeqV3(nn.Module):
    """Qwen3-Embedding -> bidirectional adapter -> Qwen3 decoder.

    Extends LLM2Seq-v2 with:
    - Source-summary alignment contrastive learning (InfoNCE)
    - Label smoothing for cross-entropy loss
    - Higher default cross-gate initialization
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_config = config.get("model", {})
        dtype = torch_dtype(str(model_config.get("dtype", "float32")))
        gradient_checkpointing = bool(model_config.get("gradient_checkpointing", True))
        fallback_hidden = int(model_config.get("hidden_size", 1024))
        encoder_hidden_size = int(model_config.get("encoder_hidden_size", fallback_hidden))
        decoder_hidden_size = int(model_config.get("decoder_hidden_size", fallback_hidden))
        self.encoder = QwenEmbeddingEncoder(
            str(model_config["encoder_name"]),
            dtype,
            gradient_checkpointing,
        )
        if self.encoder.hidden_size != encoder_hidden_size:
            raise ValueError(
                f"Configured encoder hidden size {encoder_hidden_size} != checkpoint size {self.encoder.hidden_size}"
            )
        self.adapter = SummaryAdapterV2(
            encoder_hidden_size,
            decoder_hidden_size,
            config.get("adapter", {}),
        )
        expected_banks = 3 if bool(config.get("adapter", {}).get("depth_routed_memory", False)) else 1
        configured_banks = int(config.get("decoder", {}).get("memory_bank_count", 1))
        if configured_banks != expected_banks:
            raise ValueError(
                f"Adapter produces {expected_banks} memory bank(s), but decoder.memory_bank_count={configured_banks}"
            )
        self.decoder = PretrainedQwenDecoder(
            str(model_config["decoder_name"]),
            config.get("decoder", {}),
            dtype,
            gradient_checkpointing,
        )
        decoder_hidden = int(self.decoder.config.hidden_size)
        if decoder_hidden != decoder_hidden_size:
            raise ValueError(
                f"Configured decoder hidden size {decoder_hidden_size} != checkpoint size {decoder_hidden}"
            )

        # --- Contrastive learning ---
        objectives = config.get("objectives", {})
        self.salience_weight = float(objectives.get("salience_weight", 0.1))
        self.use_contrastive = bool(objectives.get("use_contrastive", True))
        self.use_prompt_alignment = self.use_contrastive and bool(objectives.get("use_prompt_alignment", True))
        self.use_source_swap = self.use_contrastive and bool(objectives.get("use_source_swap", True))
        self.contrastive_weight = float(objectives.get("contrastive_weight", 0.1)) if self.use_prompt_alignment else 0.0
        self.contrastive_temperature = float(objectives.get("contrastive_temperature", 0.07))
        self.source_swap_weight = float(objectives.get("source_swap_weight", 0.1)) if self.use_source_swap else 0.0
        self.source_swap_margin = float(objectives.get("source_swap_margin", 0.2))
        self.source_swap_temperature = float(objectives.get("source_swap_temperature", 1.0))
        self.source_swap_strategy = str(objectives.get("source_swap_strategy", "hard_in_batch"))
        self.routing_balance_weight = float(objectives.get("routing_balance_weight", 0.0))
        self.label_smoothing = float(objectives.get("label_smoothing", 0.1))

        if self.use_prompt_alignment:
            projection_size = int(objectives.get("contrastive_projection_size", 256))
            self.alignment_head: Optional[SourceAlignmentHead] = SourceAlignmentHead(
                decoder_hidden_size,
                projection_size,
                pooling=str(objectives.get("contrastive_pooling", "mean_last")),
            )
        else:
            self.alignment_head = None

        self._stage = "unconfigured"
        self._contrastive_scale = 1.0  # adjusted during training for warmup

    @property
    def lm_head(self) -> nn.Module:
        return self.decoder.lm_head

    def set_contrastive_scale(self, scale: float) -> None:
        """Set the contrastive loss scaling factor (for warmup schedule)."""
        self._contrastive_scale = max(0.0, min(1.0, float(scale)))

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
        evidence_labels: Optional[torch.Tensor] = None,
    ) -> AdapterOutput:
        hidden_states = self.encoder(input_ids, attention_mask)
        return self.adapter(
            hidden_states,
            attention_mask,
            unit_ids=unit_ids,
            evidence_labels=evidence_labels,
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
    ) -> Dict[str, torch.Tensor]:
        adapter_output = self.encode(
            input_ids,
            attention_mask,
            unit_ids=unit_ids,
            evidence_labels=evidence_labels,
        )
        decoder_states, _ = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=adapter_output.memory,
            encoder_attention_mask=adapter_output.memory_mask,
            encoder_attention_bias=adapter_output.attention_bias,
            use_cache=False,
        )
        positive_cross_residual_ratio = self.decoder.cross_residual_ratio_mean().detach()
        positive_routing_for_loss = self.decoder.memory_routing_mean_for_loss()
        positive_routing = positive_routing_for_loss.detach()
        routing_entropy = self.decoder.memory_routing_entropy().detach()
        adaptive_routing_delta = self.decoder.adaptive_routing_delta_mean().detach()
        if labels is None:
            return {"logits": self.lm_head(decoder_states)}
        supervised = labels.ne(-100)
        if not bool(supervised.any()):
            raise ValueError("A training batch must contain supervised decoder labels")
        logits = self.lm_head(decoder_states[supervised])

        # --- Label smoothing cross-entropy ---
        loss_ce = F.cross_entropy(
            logits.float(),
            labels[supervised],
            label_smoothing=self.label_smoothing,
        )

        # --- Contrastive loss ---
        loss_contrastive = decoder_states.new_zeros(())
        prompt_retrieval_accuracy = decoder_states.new_zeros(())
        measure_source_use = self.training or bool(compute_source_diagnostics)
        # Capture the differentiable route from the correct source before the
        # counterfactual decoder pass overwrites each layer's latest route.
        loss_routing_balance = decoder_states.new_zeros(())
        if measure_source_use and self.routing_balance_weight > 0.0:
            loss_routing_balance = self.decoder.routing_balance_loss()
        if self.alignment_head is not None and measure_source_use:
            prompt_states = last_prompt_states(decoder_states, labels)
            representations = self.alignment_head(
                adapter_output.memory,
                adapter_output.memory_mask,
                prompt_states,
                bank_weights=positive_routing_for_loss if adapter_output.memory.ndim == 4 else None,
            )
            loss_contrastive = info_nce_loss(
                representations["memory_repr"],
                representations["decoder_repr"],
                self.contrastive_temperature,
            )
            similarity = representations["decoder_repr"] @ representations["memory_repr"].T
            expected = torch.arange(similarity.shape[0], device=similarity.device)
            prompt_retrieval_accuracy = similarity.detach().argmax(dim=1).eq(expected).float().mean()

        # --- Counterfactual source-swap loss ---
        loss_source_swap = decoder_states.new_zeros(())
        source_swap_nll_gap = decoder_states.new_zeros(())
        source_swap_accuracy = decoder_states.new_zeros(())
        source_swap_negative_similarity = decoder_states.new_zeros(())
        batch_size = decoder_states.shape[0]
        if measure_source_use and self.source_swap_weight > 0.0 and batch_size > 1:
            if self.source_swap_strategy == "hard_in_batch":
                with torch.no_grad():
                    source_repr = source_memory_for_mining(
                        adapter_output.memory.detach(),
                        adapter_output.memory_mask,
                        bank_weights=positive_routing.detach() if adapter_output.memory.ndim == 4 else None,
                    )
                    permutation, selected_similarity = hard_negative_indices(source_repr)
                    source_swap_negative_similarity = selected_similarity.mean().to(decoder_states.dtype)
            elif self.source_swap_strategy == "cyclic":
                # Deterministic derangement retained as the clean ablation.
                permutation = torch.arange(batch_size, device=decoder_states.device).roll(1)
            else:  # validated at config load; keep a local defensive guard.
                raise ValueError(f"Unknown source-swap strategy: {self.source_swap_strategy}")
            negative_memory = adapter_output.memory.index_select(0, permutation)
            negative_memory_mask = adapter_output.memory_mask.index_select(0, permutation)
            negative_attention_bias = (
                adapter_output.attention_bias.index_select(0, permutation)
                if adapter_output.attention_bias is not None
                else None
            )
            negative_states, _ = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=negative_memory,
                encoder_attention_mask=negative_memory_mask,
                encoder_attention_bias=negative_attention_bias,
                use_cache=False,
            )
            negative_logits = self.lm_head(negative_states[supervised])
            positive_nll = per_example_nll(logits, labels, supervised)
            negative_nll = per_example_nll(negative_logits, labels, supervised)
            loss_source_swap = source_swap_contrastive_loss(
                positive_nll,
                negative_nll,
                margin=self.source_swap_margin,
                temperature=self.source_swap_temperature,
            )
            nll_gap = negative_nll - positive_nll
            source_swap_nll_gap = nll_gap.detach().mean()
            source_swap_accuracy = nll_gap.detach().gt(0.0).float().mean()

        # --- Total loss ---
        effective_contrastive_weight = self.contrastive_weight * self._contrastive_scale
        auxiliary_scale = 1.0 if self.training else 0.0
        loss = (
            loss_ce
            + self.salience_weight * adapter_output.loss_salience.float()
            + auxiliary_scale * effective_contrastive_weight * loss_contrastive.float()
            + auxiliary_scale * self.source_swap_weight * self._contrastive_scale * loss_source_swap.float()
            + auxiliary_scale * self.routing_balance_weight * self._contrastive_scale * loss_routing_balance.float()
        )

        result = {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_salience": adapter_output.loss_salience,
            "loss_contrastive": loss_contrastive,
            "prompt_retrieval_accuracy": prompt_retrieval_accuracy,
            "loss_source_swap": loss_source_swap,
            "source_swap_nll_gap": source_swap_nll_gap,
            "source_swap_accuracy": source_swap_accuracy,
            "source_swap_negative_similarity": source_swap_negative_similarity,
            "loss_routing_balance": loss_routing_balance,
            "memory_routing_entropy": routing_entropy,
            "adaptive_routing_delta": adaptive_routing_delta,
            "contrastive_scale": decoder_states.new_tensor(self._contrastive_scale),
            "cross_gate_mean": self.decoder.cross_gate_mean().detach(),
            "cross_residual_ratio": positive_cross_residual_ratio,
            "bidirectional_gate_mean": self.adapter.bidirectional_gate_mean().detach(),
            "branch_context_gate_mean": self.adapter.branch_context_gate_mean().detach(),
            "projection_gate": torch.tanh(self.adapter.projection.residual_gate.float()).detach(),
        }
        routing = positive_routing.detach()
        if routing.numel() == 1:
            result["memory_route_summary"] = routing[0]
        else:
            for index, name in enumerate(("lexical", "semantic", "summary")):
                result[f"memory_route_{name}"] = routing[index]
        if self.adapter.salience_attention_gate is not None:
            result["salience_attention_gate"] = torch.tanh(self.adapter.salience_attention_gate.float()).detach()
        if self.alignment_head is not None and self.alignment_head.pool_gate is not None:
            result["alignment_last_pool_weight"] = torch.sigmoid(self.alignment_head.pool_gate.float()).mean().detach()
        if adapter_output.layer_weights is not None:
            for index, value in enumerate(adapter_output.layer_weights):
                result[f"fusion_weight_{index}"] = value.detach()
        if adapter_output.salience_logits is not None and evidence_labels is not None:
            width = min(adapter_output.salience_logits.shape[1], evidence_labels.shape[1])
            valid = evidence_labels[:, :width].ge(0)
            if bool(valid.any()):
                probabilities = torch.sigmoid(adapter_output.salience_logits[:, :width][valid].float())
                predictions = probabilities.ge(0.5)
                gold = evidence_labels[:, :width][valid].gt(0.5)
                true_positive = (predictions & gold).sum().float()
                result["salience_probability_mean"] = probabilities.mean().detach()
                result["salience_predicted_positive_rate"] = predictions.float().mean().detach()
                result["salience_precision"] = (true_positive / predictions.sum().float().clamp_min(1.0)).detach()
                result["salience_recall"] = (true_positive / gold.sum().float().clamp_min(1.0)).detach()
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
            frozen = [
                name
                for name, parameter in self.named_parameters()
                if parameter.numel() > 0 and not parameter.requires_grad
            ]
            if frozen:
                raise RuntimeError("Full fine-tuning left parameters frozen: " + ", ".join(frozen[:20]))

    def parameter_summary(self) -> Dict[str, int]:
        alignment_params = (
            sum(parameter.numel() for parameter in self.alignment_head.parameters())
            if self.alignment_head is not None
            else 0
        )
        return {
            "encoder": sum(parameter.numel() for parameter in self.encoder.parameters()),
            "adapter": sum(parameter.numel() for parameter in self.adapter.parameters()),
            "alignment_head": alignment_params,
            "decoder": sum(parameter.numel() for parameter in self.decoder.parameters()),
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
        }
