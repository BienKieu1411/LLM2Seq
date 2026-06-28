"""
Encoder Wrapper for LLM2Seq.

Wraps any HuggingFace causal LM or LLM2Vec model to extract
token-level hidden states for cross-attention in the decoder.

Key design:
- Returns token-level representations, NOT sentence embeddings.
- Supports output_hidden_states=True for layer fusion in the adaptor.
- Supports freeze / LoRA fine-tuning via config flags.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class EncoderWrapper(nn.Module):
    """
    Wraps a pretrained LLM backbone as a bidirectional/causal encoder.

    The wrapper extracts token-level hidden states from one or more layers,
    making them available for downstream cross-attention in the decoder.

    Args:
        model_name: HuggingFace model name or local path.
        trainable: Whether encoder parameters are trainable.
        use_lora: Whether to apply LoRA adapters (requires peft).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout.
    """

    def __init__(
        self,
        model_name: str,
        trainable: bool = False,
        use_lora: bool = False,
        torch_dtype: str = "auto",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.trainable = trainable
        self.use_lora = use_lora

        # Load the base model
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        load_kwargs = self._build_load_kwargs(torch_dtype)
        self.model = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            trust_remote_code=True,
            **load_kwargs,
        )

        # Determine hidden size from config
        self.hidden_size = getattr(self.config, "hidden_size", None)
        if self.hidden_size is None:
            raise ValueError(
                f"Cannot determine hidden_size from config of '{model_name}'. "
                "Please ensure the model config has a `hidden_size` attribute."
            )

        # Apply LoRA if requested. PEFT freezes the base model and leaves LoRA
        # tensors trainable by default; phase 3 can then freeze all tensors
        # after the adapter has been instantiated for checkpoint loading.
        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)

        # Freeze if not trainable
        if not trainable:
            self._freeze_all()

    def _build_load_kwargs(self, torch_dtype: str) -> Dict[str, torch.dtype]:
        """Build dtype kwargs for HuggingFace loading."""
        if torch_dtype in (None, "auto"):
            if torch.cuda.is_available():
                return {"torch_dtype": torch.float16}
            return {}
        if torch_dtype == "float16":
            return {"torch_dtype": torch.float16}
        if torch_dtype == "bfloat16":
            return {"torch_dtype": torch.bfloat16}
        if torch_dtype == "float32":
            return {"torch_dtype": torch.float32}
        raise ValueError(f"Unknown encoder torch_dtype: {torch_dtype}")

    def _freeze_all(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_all(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def _apply_lora(
        self,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Apply LoRA adapters to the encoder using peft."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError("peft is required for LoRA fine-tuning. Install it with: pip install peft")

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            input_ids: [batch_size, seq_len] — tokenized source input.
            attention_mask: [batch_size, seq_len] — attention mask (1 = attend, 0 = pad).
            output_hidden_states: If True, return all layer hidden states.

        Returns:
            Dict with:
                - "last_hidden_state": [B, S, d_enc] — last layer token representations.
                - "hidden_states": Tuple of [B, S, d_enc] — all layer hidden states
                  (only if output_hidden_states=True).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        result = {
            "last_hidden_state": outputs.last_hidden_state,
        }

        if output_hidden_states and hasattr(outputs, "hidden_states"):
            result["hidden_states"] = outputs.hidden_states

        return result

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers in the encoder."""
        num = getattr(self.config, "num_hidden_layers", None)
        return num if num is not None else 0

    @property
    def device(self) -> torch.device:
        """Return the device of the encoder parameters."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the encoder parameters."""
        return next(self.model.parameters()).dtype
