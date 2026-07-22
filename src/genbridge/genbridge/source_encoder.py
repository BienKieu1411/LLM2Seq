"""Unmodified causal Qwen3 used as a source feature extractor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import load_text_causal_lm, torch_dtype


@dataclass
class SourceEncoderOutput:
    """Causal token states and output-oriented suffix planning states."""

    token_states: torch.Tensor
    plan_states: torch.Tensor


class CausalSourceEncoder(nn.Module):
    """Keep Qwen causal and append trainable summary-planning embeddings.

    The source tokens keep Qwen's pretrained causal computation. Learned plan
    embeddings are appended after the source through ``inputs_embeds``; each of
    them can therefore read the complete source without changing Qwen's mask.
    Bidirectionality is introduced later by :class:`SummaryBridge`.
    """

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__()
        self.model_name = str(model_config.get("encoder_name", "Qwen/Qwen3-0.6B"))
        causal_lm, self.config = load_text_causal_lm(
            self.model_name,
            dtype=torch_dtype(str(model_config.get("dtype", "bfloat16"))),
            attn_implementation=str(model_config.get("attn_implementation", "sdpa")),
        )
        self.model = causal_lm.model
        self.hidden_size = int(self.config.hidden_size)
        self.num_layers = int(self.config.num_hidden_layers)
        self.num_summary_tokens = int(model_config.get("num_summary_tokens", 16))
        if self.num_summary_tokens < 1:
            raise ValueError("model.num_summary_tokens must be positive")
        self.summary_tokens = nn.Parameter(
            torch.empty(self.num_summary_tokens, self.hidden_size)
        )
        nn.init.normal_(
            self.summary_tokens,
            mean=0.0,
            std=float(getattr(self.config, "initializer_range", 0.02)),
        )

        use_lora = bool(model_config.get("use_lora", False))
        self.train_base = bool(model_config.get("train_base", True))
        if use_lora:
            raise ValueError("GenBridge is full-fine-tuning only; model.use_lora must be false")
        if not self.train_base:
            raise ValueError("GenBridge full fine-tuning requires model.train_base=true")

        fusion = model_config.get("layer_fusion", {}) or {}
        self.use_layer_fusion = bool(fusion.get("enabled", False))
        if self.use_layer_fusion:
            requested = list(fusion.get("indices", [-1, -5, -9, -13]))
            self.fusion_indices = self._resolve_indices(requested)
            self.fusion_logits = nn.Parameter(
                torch.zeros(len(self.fusion_indices), dtype=torch.float32)
            )
        else:
            self.fusion_indices = (self.num_layers,)
            self.register_parameter("fusion_logits", None)

        if bool(model_config.get("gradient_checkpointing", True)):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

    def _resolve_indices(self, indices: list[int]) -> Tuple[int, ...]:
        total = self.num_layers + 1
        resolved = tuple(index if index >= 0 else total + index for index in indices)
        if any(index < 0 or index >= total for index in resolved):
            raise ValueError(f"Invalid layer-fusion indices {indices} for {self.num_layers} layers")
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"Layer-fusion indices contain duplicates: {indices}")
        return resolved

    def set_backbone_trainable(self, enabled: bool) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = bool(enabled)
        # The suffix plan is part of the new interface and always trains.
        self.summary_tokens.requires_grad = True
        if self.fusion_logits is not None:
            self.fusion_logits.requires_grad = True

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_input_embeddings"):
            embedding = self.model.get_input_embeddings()
        else:  # pragma: no cover - current Qwen classes expose the method
            embedding = self.model.embed_tokens
        return embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> SourceEncoderOutput:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        token_embeddings = self._input_embeddings(input_ids)
        plans = self.summary_tokens.to(token_embeddings.dtype).unsqueeze(0).expand(
            input_ids.shape[0], -1, -1
        )
        inputs_embeds = torch.cat([token_embeddings, plans], dim=1)
        plan_mask = torch.ones(
            input_ids.shape[0],
            self.num_summary_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_mask = torch.cat([attention_mask, plan_mask], dim=1)
        # Qwen3 otherwise assigns absolute positions ``0..sequence_length-1``
        # even to a left-padded batch.  That makes the RoPE positions of both
        # real source tokens and appended plan tokens depend on the longest
        # example in the batch.  Compact positions keep an example invariant
        # to collator padding while preserving the causal order.
        position_ids = extended_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(extended_mask.eq(0), 0)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=self.use_layer_fusion,
            return_dict=True,
        )
        if self.use_layer_fusion:
            weights = torch.softmax(self.fusion_logits.float(), dim=0).to(
                outputs.last_hidden_state.dtype
            )
            selected = torch.stack(
                [outputs.hidden_states[index] for index in self.fusion_indices], dim=0
            )
            hidden_states = (weights[:, None, None, None] * selected).sum(dim=0)
        else:
            hidden_states = outputs.last_hidden_state
        source_length = input_ids.shape[1]
        return SourceEncoderOutput(
            token_states=hidden_states[:, :source_length],
            plan_states=hidden_states[:, source_length:],
        )
