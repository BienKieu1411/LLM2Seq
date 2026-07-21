"""GenBridge encoder-decoder summarizer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bridge import SummaryBridge
from .pretrained_decoder import PretrainedQwenDecoder
from .source_encoder import CausalSourceEncoder


class GenBridgeSeq2Seq(nn.Module):
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
        self.bridge = SummaryBridge(
            encoder_size=self.encoder.hidden_size,
            decoder_size=self.decoder.hidden_size,
            config=bridge_config,
        )
        reference = next(self.decoder.parameters())
        self.bridge.to(device=reference.device, dtype=reference.dtype)
        objectives = config.get("objectives", {}) or {}
        self.salience_weight = float(objectives.get("salience_weight", 0.2))
        self.plan_alignment_weight = float(objectives.get("plan_alignment_weight", 0.1))
        self.plan_diversity_weight = float(objectives.get("plan_diversity_weight", 0.01))

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
        encoder_output = self.encoder(input_ids, attention_mask)
        bridge_output = self.bridge(
            encoder_output.token_states,
            encoder_output.plan_states,
            attention_mask,
            unit_ids,
            evidence_labels=evidence_labels,
        )
        if return_bridge_output:
            return bridge_output
        if return_attention_mask:
            return bridge_output.memory, bridge_output.memory_mask
        return bridge_output.memory

    def decoder_memory_kwargs(self, bridge_output: Any) -> Dict[str, torch.Tensor]:
        """Build either the proposed dual-memory or concatenation interface."""

        memory_kwargs = {
            "encoder_hidden_states": bridge_output.memory,
            "encoder_attention_mask": bridge_output.memory_mask,
        }
        if self.bridge.mode == "genbridge" and self.decoder.memory_attention == "gated_dual":
            memory_kwargs.update(
                {
                    "token_encoder_hidden_states": bridge_output.token_memory,
                    "token_encoder_attention_mask": bridge_output.token_memory_mask,
                    "plan_encoder_hidden_states": bridge_output.plan_memory,
                    "plan_encoder_attention_mask": bridge_output.plan_memory_mask,
                }
            )
        return memory_kwargs

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
            attention_mask=decoder_attention_mask,
            **self.decoder_memory_kwargs(bridge_output),
        )
        if labels is None:
            return {"logits": self.lm_head(decoder_states)}

        supervised = labels.ne(-100)
        if not bool(supervised.any()):
            raise ValueError("At least one decoder label must be supervised")
        selected_logits = self.lm_head(decoder_states[supervised])
        loss_ce = F.cross_entropy(selected_logits.float(), labels[supervised])
        target_mask = labels.ne(-100)
        target_state = (
            decoder_states.float() * target_mask.unsqueeze(-1)
        ).sum(dim=1) / target_mask.sum(dim=1, keepdim=True).clamp_min(1)
        plan_state = bridge_output.plan_memory.float().mean(dim=1)
        loss_plan_alignment = 1.0 - F.cosine_similarity(
            plan_state,
            target_state.detach(),
            dim=-1,
        ).mean()
        if not self.bridge.use_plan_alignment:
            loss_plan_alignment = loss_plan_alignment.detach() * 0.0
        loss = (
            loss_ce
            + self.salience_weight * bridge_output.loss_salience.float()
            + self.plan_alignment_weight * loss_plan_alignment.float()
            + self.plan_diversity_weight * bridge_output.loss_plan_diversity.float()
        )
        return {
            "logits": selected_logits,
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_salience": bridge_output.loss_salience,
            "loss_plan_alignment": loss_plan_alignment,
            "loss_plan_diversity": bridge_output.loss_plan_diversity,
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
        if full:
            frozen = [name for name, parameter in self.named_parameters() if not parameter.requires_grad]
            if frozen:
                raise RuntimeError(
                    "Full-finetune stage left parameters frozen: " + ", ".join(frozen[:20])
                )

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def summary(self) -> str:
        total = sum(parameter.numel() for parameter in self.parameters())
        return "\n".join(
            [
                "GenBridgeSeq2Seq",
                f"  causal source encoder: {self.encoder.model_name}",
                f"  bridge mode: {self.bridge.mode}",
                f"  summary planning tokens: {self.encoder.num_summary_tokens}",
                f"  bidirectional token adapter: {self.bridge.token_encoder is not None}",
                f"  decoder layers copied from: {list(self.decoder.layer_indices)}",
                f"  decoder cross-attention layers: {list(self.decoder.cross_attention_indices)}",
                f"  decoder memory attention: {self.decoder.memory_attention}",
                f"  total parameters: {total:,}",
                f"  trainable parameters: {self.trainable_parameters():,}",
            ]
        )
