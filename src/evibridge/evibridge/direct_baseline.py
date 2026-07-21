"""Direct causal Qwen LoRA baseline used by the controlled comparison."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import load_text_causal_lm, torch_dtype


class DirectCausalBaseline(nn.Module):
    """Direct causal Qwen baseline under the same full-FT/LoRA regime."""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__()
        self.model_name = str(model_config.get("encoder_name", "Qwen/Qwen3.5-0.8B"))
        self.model, _ = load_text_causal_lm(
            self.model_name,
            dtype=torch_dtype(str(model_config.get("dtype", "bfloat16"))),
            attn_implementation="sdpa",
        )
        use_lora = bool(model_config.get("use_lora", False))
        train_base = bool(model_config.get("train_base", True))
        if use_lora and train_base:
            raise ValueError("Choose either full fine-tuning or LoRA for the direct baseline")
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            lora = model_config.get("lora", {}) or {}
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(lora.get("r", 32)),
                lora_alpha=int(lora.get("alpha", 64)),
                lora_dropout=float(lora.get("dropout", 0.05)),
                target_modules=list(lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
                bias="none",
            )
            self.model = get_peft_model(self.model, peft_config)
        elif not train_base:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        if bool(model_config.get("gradient_checkpointing", True)):
            self.model.gradient_checkpointing_enable()
            if use_lora:
                self.model.enable_input_require_grads()
        self.model.config.use_cache = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        causal_lm = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        output = causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        if labels is None:
            logits = causal_lm.lm_head(output.last_hidden_state)
            return {"logits": logits}

        # Only summary positions carry labels. Projecting every 3k-token prompt
        # to Qwen3.5's 248k vocabulary would waste tens of GB at physical batch
        # 32, while contributing exactly zero supervised loss.
        shift_labels = labels[:, 1:].contiguous()
        shift_hidden = output.last_hidden_state[:, :-1, :]
        supervised = shift_labels.ne(-100)
        if not bool(supervised.any()):
            raise ValueError("Direct causal batch contains no supervised target token")
        selected_logits = causal_lm.lm_head(shift_hidden[supervised])
        selected_labels = shift_labels[supervised]
        loss = F.cross_entropy(selected_logits.float(), selected_labels)
        result = {"logits": selected_logits, "loss": loss, "loss_ce": loss}
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
