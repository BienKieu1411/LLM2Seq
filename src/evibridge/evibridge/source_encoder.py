"""Unmodified causal LLM used as the source feature extractor."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import load_text_causal_lm, torch_dtype


class CausalSourceEncoder(nn.Module):
    """Preserve the pretrained causal computation and optionally fuse layers."""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__()
        self.model_name = str(model_config.get("encoder_name", "Qwen/Qwen3.5-0.8B"))
        causal_lm, self.config = load_text_causal_lm(
            self.model_name,
            dtype=torch_dtype(str(model_config.get("dtype", "bfloat16"))),
            attn_implementation="sdpa",
        )
        self.model = causal_lm.model
        self.hidden_size = int(self.config.hidden_size)
        self.num_layers = int(self.config.num_hidden_layers)
        self.use_lora = bool(model_config.get("use_lora", False))
        self.train_base = bool(model_config.get("train_base", True))
        if self.use_lora and self.train_base:
            raise ValueError("Choose either full fine-tuning or LoRA for the source encoder")
        if self.use_lora:
            self.model = self._apply_lora(model_config)
        elif not self.train_base:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        fusion = model_config.get("layer_fusion", {}) or {}
        self.use_layer_fusion = bool(fusion.get("enabled", False))
        if self.use_layer_fusion:
            requested = list(fusion.get("indices", [-1, -5, -9, -13]))
            self.fusion_indices = self._resolve_indices(requested)
            self.fusion_logits = nn.Parameter(torch.zeros(len(self.fusion_indices), dtype=torch.float32))
        else:
            self.fusion_indices = (self.num_layers,)
            self.register_parameter("fusion_logits", None)

        if bool(model_config.get("gradient_checkpointing", True)):
            self.model.gradient_checkpointing_enable()
            if self.use_lora and hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

    def _resolve_indices(self, indices: list[int]) -> Tuple[int, ...]:
        # hidden_states contains the embedding output followed by N layer outputs.
        total = self.num_layers + 1
        resolved = tuple(index if index >= 0 else total + index for index in indices)
        if any(index < 0 or index >= total for index in resolved):
            raise ValueError(f"Invalid layer-fusion indices {indices} for {self.num_layers} layers")
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"Layer-fusion indices contain duplicates: {indices}")
        return resolved

    def _apply_lora(self, model_config: Dict[str, Any]) -> nn.Module:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("PEFT is required when model.use_lora=true") from exc
        lora = model_config.get("lora", {}) or {}
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(lora.get("r", 32)),
            lora_alpha=int(lora.get("alpha", 64)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            bias="none",
        )
        return get_peft_model(self.model, peft_config)

    def set_backbone_trainable(self, enabled: bool) -> None:
        if self.train_base:
            for parameter in self.model.parameters():
                parameter.requires_grad = bool(enabled)
        elif self.use_lora:
            for name, parameter in self.model.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad = bool(enabled)
        if self.fusion_logits is not None:
            self.fusion_logits.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=self.use_layer_fusion,
            return_dict=True,
        )
        if not self.use_layer_fusion:
            return outputs.last_hidden_state
        weights = torch.softmax(self.fusion_logits.float(), dim=0).to(outputs.last_hidden_state.dtype)
        selected = torch.stack([outputs.hidden_states[index] for index in self.fusion_indices], dim=0)
        return (weights[:, None, None, None] * selected).sum(dim=0)
