"""Qwen3-Base encoder with layer-wise routed source attention."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .mask_policy import LayerMaskPolicy, MaskPolicyConfig
from .routed_attention import RoutedSelfAttention


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(mapping)}")
    return mapping[name]


def _find_decoder_layers(model: nn.Module, expected_layers: int) -> nn.ModuleList:
    candidates: List[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if not isinstance(layers, nn.ModuleList) or len(layers) != expected_layers:
            continue
        if all(hasattr(layer, "self_attn") for layer in layers):
            candidates.append((name, layers))
    if not candidates:
        raise RuntimeError(
            f"Could not locate {expected_layers} Qwen-style decoder layers. "
            "AdaBiMask currently expects layers exposing a self_attn module."
        )
    # The shortest path is the backbone container rather than an accidental
    # nested match, and is stable before/after PEFT wrapping.
    candidates.sort(key=lambda item: (item[0].count("."), len(item[0])))
    return candidates[0][1]


class AdaBiMaskEncoder(nn.Module):
    """Load a causal Qwen3 base model and route its attention masks by layer."""

    def __init__(self, model_config: Dict[str, Any], mask_config: Dict[str, Any]):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError("Install src/adabimask/requirements.txt before building the encoder") from exc

        model_name = str(model_config.get("encoder_name", "Qwen/Qwen3-0.6B-Base"))
        dtype = _torch_dtype(str(model_config.get("dtype", "bfloat16")))
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = int(getattr(self.config, "hidden_size"))
        self.num_layers = int(getattr(self.config, "num_hidden_layers"))

        # SDPA accepts a 4-D additive padding-only mask. FlashAttention-2's
        # Transformers interface expects a 2-D mask and cannot be switched per
        # layer by this wrapper.
        self.model = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        use_lora = bool(model_config.get("use_lora", True))
        train_base = bool(model_config.get("train_base", False))
        if train_base:
            raise ValueError(
                "Full backbone training is intentionally disabled: controlled AdaBiMask experiments use LoRA only"
            )
        if use_lora:
            self.model = self._apply_lora(model_config)
        else:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        self.policy = LayerMaskPolicy(self.num_layers, MaskPolicyConfig.from_dict(mask_config))
        layers = _find_decoder_layers(self.model, self.num_layers)
        for layer_index, layer in enumerate(layers):
            layer.self_attn = RoutedSelfAttention(layer.self_attn, layer_index, self.policy)

        if bool(model_config.get("gradient_checkpointing", True)):
            self.model.gradient_checkpointing_enable()
            if use_lora and hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

    def _apply_lora(self, model_config: Dict[str, Any]) -> nn.Module:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("PEFT is required when model.use_lora=true") from exc

        lora = model_config.get("lora", {}) or {}
        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            bias="none",
        )
        return get_peft_model(self.model, config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        result: Dict[str, Any] = {"last_hidden_state": outputs.last_hidden_state}
        if output_hidden_states:
            result["hidden_states"] = outputs.hidden_states
        return result

    def gate_regularization(self) -> Dict[str, torch.Tensor]:
        return self.policy.regularization()

    def policy_state(self) -> Dict[str, object]:
        return self.policy.describe()
