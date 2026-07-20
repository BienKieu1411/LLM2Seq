"""AdaBiMask encoder-decoder summarizer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import AdaBiMaskEncoder

try:
    # The sibling llm2seq project owns the already-tested scratch decoder. The
    # supplied run.sh exposes it without changing the legacy project.
    from src.models.decoder import LightweightDecoder
except ImportError as exc:  # pragma: no cover - exercised by environment smoke check
    raise ImportError(
        "Cannot import the sibling llm2seq scratch decoder. Run through src/adabimask/run.sh "
        "or add both src/adabimask and src/llm2seq to PYTHONPATH."
    ) from exc


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
    """Qwen3-Base source encoder, minimal projection, and scratch decoder."""

    def __init__(self, config: Dict[str, Any], vocab_size: int):
        super().__init__()
        self.raw_config = config
        model_config = config.get("model", {}) or {}
        decoder_config = config.get("decoder", {}) or {}
        self.encoder = AdaBiMaskEncoder(model_config, config.get("mask", {}) or {})

        decoder_size = int(decoder_config.get("hidden_size", self.encoder.hidden_size))
        self.adaptor = MemoryProjection(
            encoder_size=self.encoder.hidden_size,
            decoder_size=decoder_size,
            dropout=float(decoder_config.get("dropout", 0.1)),
        )
        self.decoder = LightweightDecoder(
            vocab_size=vocab_size,
            hidden_size=decoder_size,
            num_layers=int(decoder_config.get("num_layers", 8)),
            num_heads=int(decoder_config.get("num_heads", 16)),
            ffn_size=int(decoder_config.get("ffn_size", 4096)),
            max_seq_len=int(decoder_config.get("max_target_length", 512)),
            dropout=float(decoder_config.get("dropout", 0.1)),
            tie_embeddings=bool(decoder_config.get("tie_embeddings", True)),
        )
        self.lm_head = nn.Linear(decoder_size, vocab_size, bias=False)
        if bool(decoder_config.get("tie_embeddings", True)):
            self.lm_head.weight = self.decoder.embed_tokens.weight

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
        logits = self.lm_head(decoder_states)
        result: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
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
                f"  total parameters: {total:,}",
                f"  trainable parameters: {self.trainable_parameters():,}",
            ]
        )
