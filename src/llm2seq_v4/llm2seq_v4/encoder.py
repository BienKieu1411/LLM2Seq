"""Generic token encoder for pretrained text-embedding checkpoints."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _config_is_bidirectional(config: object) -> bool:
    """Recognize the explicit flags used by supported embedding checkpoints."""

    if bool(getattr(config, "use_bidirectional_attention", False)):
        return True
    is_causal = getattr(config, "is_causal", None)
    return is_causal is False


class EmbeddingTokenEncoder(nn.Module):
    """Expose all token hidden states from an ``AutoModel`` embedding backbone."""

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype,
        gradient_checkpointing: bool,
        *,
        attn_implementation: str = "sdpa",
        expected_hidden_layers: int | None = None,
        expected_attention_mode: str = "auto",
        trust_remote_code: bool = True,
        revision: str | None = None,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        pretrained_kwargs = {
            "trust_remote_code": bool(trust_remote_code),
        }
        if revision is not None:
            pretrained_kwargs["revision"] = str(revision)
        config = AutoConfig.from_pretrained(model_name, **pretrained_kwargs)
        self.model = AutoModel.from_pretrained(
            model_name,
            config=config,
            dtype=dtype,
            attn_implementation=attn_implementation,
            **pretrained_kwargs,
        )
        self.config = config
        self.model_name = str(model_name)
        self.revision = str(revision) if revision is not None else None
        self.trust_remote_code = bool(trust_remote_code)
        self.hidden_size = int(config.hidden_size)
        self.num_hidden_layers = int(getattr(config, "num_hidden_layers", 0))
        if expected_hidden_layers is not None and self.num_hidden_layers != int(expected_hidden_layers):
            raise ValueError(
                f"Configured encoder layer count {expected_hidden_layers} != checkpoint size {self.num_hidden_layers}"
            )
        expected_attention_mode = str(expected_attention_mode).lower()
        if expected_attention_mode not in {"auto", "causal", "bidirectional"}:
            raise ValueError(f"Unsupported encoder attention mode {expected_attention_mode!r}")
        self.is_bidirectional = _config_is_bidirectional(config)
        if expected_attention_mode == "bidirectional" and not self.is_bidirectional:
            raise ValueError(
                f"Encoder {model_name!r} is expected to be bidirectional, but its config exposes "
                "neither use_bidirectional_attention=true nor is_causal=false"
            )
        if expected_attention_mode == "causal" and self.is_bidirectional:
            raise ValueError(f"Encoder {model_name!r} is configured as bidirectional, not causal")
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
            raise RuntimeError(f"Embedding encoder {self.model_name!r} did not return token hidden states")
        hidden_states = tuple(outputs.hidden_states)
        expected = self.num_hidden_layers + 1
        if self.num_hidden_layers > 0 and len(hidden_states) != expected:
            raise RuntimeError(
                f"Embedding encoder {self.model_name!r} returned {len(hidden_states)} hidden states; "
                f"expected embedding output plus {self.num_hidden_layers} layers ({expected})"
            )
        return hidden_states

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = bool(trainable)


# Backwards-compatible import for older checkpoints/scripts. The implementation
# is intentionally no longer Qwen-specific.
QwenEmbeddingEncoder = EmbeddingTokenEncoder
