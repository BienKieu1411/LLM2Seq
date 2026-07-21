"""Backbone loading helpers shared by the encoder, decoder, and baseline."""

from __future__ import annotations

from typing import Any, Tuple

import torch


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(mapping)}")
    return mapping[name]


def load_text_causal_lm(
    model_name: str,
    dtype: torch.dtype,
    attn_implementation: str = "sdpa",
) -> Tuple[Any, Any]:
    """Load only the language model from text or multimodal Qwen checkpoints.

    Qwen3.5 checkpoints are distributed with a multimodal top-level config. Passing the
    text sub-config to AutoModelForCausalLM activates Transformers' official
    checkpoint prefix conversion and avoids materializing the vision tower.
    """

    from transformers import AutoConfig, AutoModelForCausalLM

    raw_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_config = raw_config.get_text_config() if hasattr(raw_config, "get_text_config") else raw_config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=text_config,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    return model, text_config
