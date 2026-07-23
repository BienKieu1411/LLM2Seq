"""Qwen3-Embedding token encoder used by LLM2Seq-v2."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class QwenEmbeddingEncoder(nn.Module):
    def __init__(self, model_name: str, dtype: torch.dtype, gradient_checkpointing: bool):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            config=config,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.config = config
        self.model_name = str(model_name)
        self.hidden_size = int(config.hidden_size)
        self.model.config.use_cache = False
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("Qwen embedding encoder did not return hidden states")
        return tuple(outputs.hidden_states)

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = bool(trainable)
