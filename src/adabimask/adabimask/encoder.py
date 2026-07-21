"""Qwen3/Qwen3.5 encoder with layer-wise routed source token mixers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .backbone import load_text_causal_lm, torch_dtype
from .mask_policy import LayerMaskPolicy, MaskPolicyConfig
from .routed_attention import RoutedLinearAttention, RoutedSelfAttention


def _find_decoder_layers(model: nn.Module, expected_layers: int) -> nn.ModuleList:
    candidates: List[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if not isinstance(layers, nn.ModuleList) or len(layers) != expected_layers:
            continue
        if all(hasattr(layer, "self_attn") or hasattr(layer, "linear_attn") for layer in layers):
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
    """Load a causal Qwen text model and route source context by layer."""

    def __init__(self, model_config: Dict[str, Any], mask_config: Dict[str, Any]):
        super().__init__()
        model_name = str(model_config.get("encoder_name", "Qwen/Qwen3.5-0.8B"))
        dtype = torch_dtype(str(model_config.get("dtype", "bfloat16")))
        self.model_name = model_name
        causal_lm, self.config = load_text_causal_lm(model_name, dtype=dtype, attn_implementation="sdpa")
        self.hidden_size = int(getattr(self.config, "hidden_size"))
        self.num_layers = int(getattr(self.config, "num_hidden_layers"))
        # Keep only the text backbone. The LM head is not needed by the source
        # encoder and is tied to the embedding in Qwen checkpoints anyway.
        self.model = causal_lm.model

        use_lora = bool(model_config.get("use_lora", True))
        train_base = bool(model_config.get("train_base", False))
        if train_base and use_lora:
            raise ValueError("Choose either full fine-tuning (train_base=true) or LoRA, not both")
        self.use_lora = use_lora
        self.train_base = train_base
        if use_lora:
            self.model = self._apply_lora(model_config)
        elif not train_base:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        self.policy = LayerMaskPolicy(self.num_layers, MaskPolicyConfig.from_dict(mask_config))
        layers = _find_decoder_layers(self.model, self.num_layers)
        for layer_index, layer in enumerate(layers):
            if hasattr(layer, "self_attn"):
                layer.self_attn = RoutedSelfAttention(layer.self_attn, layer_index, self.policy)
            elif hasattr(layer, "linear_attn"):
                layer.linear_attn = RoutedLinearAttention(layer.linear_attn, layer_index, self.policy)
            else:  # pragma: no cover - guarded by _find_decoder_layers
                raise RuntimeError(f"Layer {layer_index} has no supported token mixer")

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

    def set_backbone_trainable(self, enabled: bool) -> None:
        """Toggle pretrained encoder updates for staged training."""

        if self.train_base:
            for parameter in self.model.parameters():
                parameter.requires_grad = bool(enabled)
        elif self.use_lora:
            for name, parameter in self.model.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad = bool(enabled)

    def set_curriculum_progress(self, progress: float) -> None:
        self.policy.set_progress(progress)

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
