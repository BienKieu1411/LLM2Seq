"""The faithful LLM2Seq-v2 encoder-adapter-decoder model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import AdapterOutput, SummaryAdapterV2
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


class LLM2SeqV2(nn.Module):
    """Qwen3-Embedding -> bidirectional adapter -> Qwen3 decoder."""

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
        self.salience_weight = float(config.get("objectives", {}).get("salience_weight", 0.1))
        self._stage = "unconfigured"

    @property
    def lm_head(self) -> nn.Module:
        return self.decoder.lm_head

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
        if labels is None:
            return {"logits": self.lm_head(decoder_states)}
        supervised = labels.ne(-100)
        if not bool(supervised.any()):
            raise ValueError("A training batch must contain supervised decoder labels")
        logits = self.lm_head(decoder_states[supervised])
        loss_ce = F.cross_entropy(logits.float(), labels[supervised])
        loss = loss_ce + self.salience_weight * adapter_output.loss_salience.float()
        result = {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_salience": adapter_output.loss_salience,
            "cross_gate_mean": self.decoder.cross_gate_mean().detach(),
            "cross_residual_ratio": self.decoder.cross_residual_ratio_mean().detach(),
            "bidirectional_gate_mean": self.adapter.bidirectional_gate_mean().detach(),
            "projection_gate": torch.tanh(self.adapter.projection.residual_gate.float()).detach(),
        }
        if self.adapter.salience_attention_gate is not None:
            result["salience_attention_gate"] = torch.tanh(
                self.adapter.salience_attention_gate.float()
            ).detach()
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
                result["salience_precision"] = (
                    true_positive / predictions.sum().float().clamp_min(1.0)
                ).detach()
                result["salience_recall"] = (
                    true_positive / gold.sum().float().clamp_min(1.0)
                ).detach()
        return result

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"interface_warmup", "full_finetune"}:
            raise ValueError(f"Unknown training stage: {stage}")
        full = stage == "full_finetune"
        self.encoder.set_trainable(full)
        self.decoder.set_backbone_trainable(full)
        for parameter in self.adapter.parameters():
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
        return {
            "encoder": sum(parameter.numel() for parameter in self.encoder.parameters()),
            "adapter": sum(parameter.numel() for parameter in self.adapter.parameters()),
            "decoder": sum(parameter.numel() for parameter in self.decoder.parameters()),
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
        }
