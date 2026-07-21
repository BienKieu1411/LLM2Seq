"""AdaBiMask encoder-decoder summarizer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import AdaBiMaskEncoder
from .pretrained_decoder import PretrainedQwenDecoder


class MemoryProjection(nn.Module):
    """The intentionally minimal interface used in all mask ablations."""

    def __init__(self, encoder_size: int, decoder_size: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(encoder_size)
        self.projection = nn.Linear(encoder_size, decoder_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.projection(self.norm(hidden_states)))


class AdaBiMaskSeq2Seq(nn.Module):
    """AdaBiMask source encoder and a causal pretrained shallow Qwen decoder."""

    def __init__(self, config: Dict[str, Any], vocab_size: int):
        super().__init__()
        self.raw_config = config
        model_config = config.get("model", {}) or {}
        decoder_config = config.get("decoder", {}) or {}
        self.encoder = AdaBiMaskEncoder(model_config, config.get("mask", {}) or {})

        decoder_name = str(decoder_config.get("pretrained_name", self.encoder.model_name))
        self.decoder = PretrainedQwenDecoder(
            decoder_name,
            decoder_config,
            dtype_name=str(model_config.get("dtype", "bfloat16")),
        )
        if self.encoder.hidden_size == self.decoder.hidden_size:
            self.adaptor = nn.Identity()
        else:
            self.adaptor = MemoryProjection(
                encoder_size=self.encoder.hidden_size,
                decoder_size=self.decoder.hidden_size,
                dropout=float(decoder_config.get("dropout", 0.0)),
            )
            reference_parameter = next(self.decoder.parameters())
            self.adaptor.to(
                device=reference_parameter.device,
                dtype=reference_parameter.dtype,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        memory = self.encode(input_ids, attention_mask)
        decoder_states, _ = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=memory,
            encoder_attention_mask=attention_mask,
            attention_mask=decoder_attention_mask,
        )
        if labels is None:
            # Generation still needs the dense [batch, length, vocabulary]
            # projection. Training does not: padding labels are ignored by the
            # loss, so projecting them only wastes most of the large Qwen
            # vocabulary GEMM.
            logits = self.lm_head(decoder_states)
            return {"logits": logits}

        supervised = labels.ne(-100)
        selected_states = decoder_states[supervised]
        if selected_states.shape[0] == 0:
            raise ValueError("At least one decoder label must be supervised")
        selected_logits = self.lm_head(selected_states)
        selected_labels = labels[supervised]
        result: Dict[str, torch.Tensor] = {"logits": selected_logits}
        loss_ce = F.cross_entropy(selected_logits.float(), selected_labels)
        gate_losses = self.encoder.gate_regularization()
        result.update(gate_losses)
        result["loss_ce"] = loss_ce
        result["loss"] = loss_ce + gate_losses["loss_gate"]
        return result

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention_mask: bool = False,
    ) -> Any:
        encoder_output = self.encoder(input_ids, attention_mask, output_hidden_states=False)
        memory = self.adaptor(encoder_output["last_hidden_state"])
        if return_attention_mask:
            return memory, attention_mask
        return memory

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def lm_head(self) -> nn.Module:
        return self.decoder.lm_head

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"interface_warmup", "full_finetune"}:
            raise ValueError(f"Unknown training stage: {stage}")
        full = stage == "full_finetune"
        self.encoder.set_backbone_trainable(full)
        self.decoder.set_backbone_trainable(full)
        for parameter in self.adaptor.parameters():
            parameter.requires_grad = True
        if self.encoder.policy.gate_logits is not None:
            self.encoder.policy.gate_logits.requires_grad = full
        self.encoder.policy.set_force_causal(not full)
        self.encoder.set_curriculum_progress(0.0 if not full else 1e-6)

    def set_curriculum_progress(self, progress: float) -> None:
        self.encoder.set_curriculum_progress(progress)

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def summary(self) -> str:
        policy = self.encoder.policy_state()
        total = sum(parameter.numel() for parameter in self.parameters())
        return "\n".join(
            [
                "AdaBiMaskSeq2Seq",
                f"  encoder: {self.encoder.model_name}",
                f"  mask mode: {policy['mode']}",
                f"  groups: {policy['groups']}",
                f"  selected groups: {policy['selected_groups']}",
                f"  selected layers: {policy['selected_layers']}",
                f"  decoder layers copied from: {list(self.decoder.layer_indices)}",
                f"  total parameters: {total:,}",
                f"  trainable parameters: {self.trainable_parameters():,}",
            ]
        )
