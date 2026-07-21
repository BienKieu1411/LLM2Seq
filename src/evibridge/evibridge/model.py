"""EviBridge encoder-decoder summarizer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bridge import EvidenceBridge
from .pretrained_decoder import PretrainedQwenDecoder
from .source_encoder import CausalSourceEncoder


class EviBridgeSeq2Seq(nn.Module):
    """Causal source LLM, summary planner, and pretrained causal decoder."""

    def __init__(self, config: Dict[str, Any], vocab_size: int):
        super().__init__()
        del vocab_size  # The tied pretrained LM head defines the vocabulary.
        self.raw_config = config
        model_config = config.get("model", {}) or {}
        decoder_config = config.get("decoder", {}) or {}
        bridge_config = config.get("bridge", {}) or {}
        self.encoder = CausalSourceEncoder(model_config)
        decoder_name = str(decoder_config.get("pretrained_name", self.encoder.model_name))
        self.decoder = PretrainedQwenDecoder(
            decoder_name,
            decoder_config,
            dtype_name=str(model_config.get("dtype", "bfloat16")),
        )
        self.bridge = EvidenceBridge(
            encoder_size=self.encoder.hidden_size,
            decoder_size=self.decoder.hidden_size,
            config=bridge_config,
        )
        reference = next(self.decoder.parameters())
        self.bridge.to(device=reference.device, dtype=reference.dtype)
        objectives = config.get("objectives", {}) or {}
        self.evidence_weight = float(objectives.get("evidence_weight", 0.2))
        self.diversity_weight = float(objectives.get("diversity_weight", 0.01))

    @property
    def lm_head(self) -> nn.Module:
        return self.decoder.lm_head

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
        evidence_labels: Optional[torch.Tensor] = None,
        return_attention_mask: bool = False,
        return_bridge_output: bool = False,
    ) -> Any:
        if unit_ids is None:
            unit_ids = attention_mask.long()
        hidden_states = self.encoder(input_ids, attention_mask)
        bridge_output = self.bridge(
            hidden_states,
            attention_mask,
            unit_ids,
            evidence_labels=evidence_labels,
        )
        if return_bridge_output:
            return bridge_output
        if return_attention_mask:
            return bridge_output.memory, bridge_output.memory_mask
        return bridge_output.memory

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
        evidence_labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        bridge_output = self.encode(
            input_ids,
            attention_mask,
            unit_ids=unit_ids,
            evidence_labels=evidence_labels,
            return_bridge_output=True,
        )
        decoder_states, _ = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=bridge_output.memory,
            encoder_attention_mask=bridge_output.memory_mask,
            attention_mask=decoder_attention_mask,
        )
        if labels is None:
            return {"logits": self.lm_head(decoder_states)}

        supervised = labels.ne(-100)
        if not bool(supervised.any()):
            raise ValueError("At least one decoder label must be supervised")
        selected_logits = self.lm_head(decoder_states[supervised])
        loss_ce = F.cross_entropy(selected_logits.float(), labels[supervised])
        loss = (
            loss_ce
            + self.evidence_weight * bridge_output.loss_evidence.float()
            + self.diversity_weight * bridge_output.loss_diversity.float()
        )
        return {
            "logits": selected_logits,
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_evidence": bridge_output.loss_evidence,
            "loss_diversity": bridge_output.loss_diversity,
        }

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"interface_warmup", "full_finetune"}:
            raise ValueError(f"Unknown training stage: {stage}")
        full = stage == "full_finetune"
        self.encoder.set_backbone_trainable(full)
        self.decoder.set_backbone_trainable(full)
        # The summary planner and new cross-attention are always optimized.
        for parameter in self.bridge.parameters():
            parameter.requires_grad = True

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def summary(self) -> str:
        total = sum(parameter.numel() for parameter in self.parameters())
        return "\n".join(
            [
                "EviBridgeSeq2Seq",
                f"  causal source encoder: {self.encoder.model_name}",
                f"  bridge mode: {self.bridge.mode}",
                f"  evidence slots: {self.bridge.num_slots}",
                f"  decoder layers copied from: {list(self.decoder.layer_indices)}",
                f"  total parameters: {total:,}",
                f"  trainable parameters: {self.trainable_parameters():,}",
            ]
        )
