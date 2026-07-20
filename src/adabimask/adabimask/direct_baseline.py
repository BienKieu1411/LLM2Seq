"""Direct causal Qwen LoRA baseline used by the controlled comparison."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


class DirectCausalBaseline(nn.Module):
    """Qwen3-Base trained to generate a summary after the source prompt."""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM

        self.model_name = str(model_config.get("encoder_name", "Qwen/Qwen3-0.6B-Base"))
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=_torch_dtype(str(model_config.get("dtype", "bfloat16"))),
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        lora = model_config.get("lora", {}) or {}
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        if bool(model_config.get("gradient_checkpointing", True)):
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        self.model.config.use_cache = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        result = {"logits": output.logits}
        if labels is not None:
            result["loss"] = output.loss
            result["loss_ce"] = output.loss
        return result

    def generate(self, **kwargs: Any) -> torch.Tensor:
        previous_use_cache = self.model.config.use_cache
        self.model.config.use_cache = True
        try:
            return self.model.generate(**kwargs)
        finally:
            self.model.config.use_cache = previous_use_cache

    def summary(self) -> str:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return "\n".join(
            [
                "DirectCausalBaseline",
                f"  model: {self.model_name}",
                f"  total parameters: {total:,}",
                f"  trainable parameters: {trainable:,}",
            ]
        )
