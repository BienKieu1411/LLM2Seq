"""Configuration loading and validation for LLM2Seq-v2."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("_base_", None)
    if parent:
        config = _merge(load_config(path.parent / str(parent)), config)
    validate_config(config)
    config.setdefault("_meta", {})["config_path"] = str(path)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    model = config.get("model", {})
    adapter = config.get("adapter", {})
    decoder = config.get("decoder", {})
    training = config.get("training", {})
    data = config.get("data", {})

    for field in ("encoder_name", "decoder_name"):
        if not str(model.get(field, "")).strip():
            raise ValueError(f"model.{field} is required")
    fallback_hidden = int(model.get("hidden_size", 0))
    encoder_hidden = int(model.get("encoder_hidden_size", fallback_hidden))
    decoder_hidden = int(model.get("decoder_hidden_size", fallback_hidden))
    if encoder_hidden <= 0 or decoder_hidden <= 0:
        raise ValueError("model encoder/decoder hidden sizes must be positive")

    indices = list(adapter.get("fuse_layers", []))
    if bool(adapter.get("layer_fusion", True)) and not indices:
        raise ValueError("adapter.fuse_layers cannot be empty when layer fusion is enabled")
    if int(adapter.get("num_bidirectional_layers", 0)) < 0:
        raise ValueError("adapter.num_bidirectional_layers must be non-negative")
    hidden_size = int(adapter.get("hidden_size", decoder_hidden))
    heads = int(adapter.get("num_heads", 1))
    if hidden_size % heads:
        raise ValueError("adapter hidden size must be divisible by adapter.num_heads")
    depth_routed = bool(adapter.get("depth_routed_memory", False))
    expected_banks = 3 if depth_routed else 1
    configured_banks = int(decoder.get("memory_bank_count", 1))
    if configured_banks != expected_banks:
        raise ValueError(
            f"decoder.memory_bank_count must be {expected_banks} when "
            f"adapter.depth_routed_memory={depth_routed}"
        )
    if depth_routed:
        for field in ("lexical_layers", "semantic_layers"):
            if not list(adapter.get(field, [])):
                raise ValueError(f"adapter.{field} cannot be empty for depth-routed memory")
    if bool(adapter.get("hierarchical_sentence_context", False)):
        context_size = int(adapter.get("sentence_context_size", 256))
        context_heads = int(adapter.get("sentence_context_heads", 8))
        if context_size <= 0 or context_size % context_heads:
            raise ValueError(
                "adapter.sentence_context_size must be positive and divisible "
                "by adapter.sentence_context_heads"
            )

    cross_every = int(decoder.get("cross_attention_every", 1))
    if cross_every <= 0:
        raise ValueError("decoder.cross_attention_every must be positive")
    gate = float(decoder.get("cross_gate_init", 0.1))
    if not 0.0 <= gate < 1.0:
        raise ValueError("decoder.cross_gate_init must be in [0, 1)")

    warmup_epochs = int(training.get("interface_warmup_epochs", 0))
    full_epochs = int(training.get("full_finetune_epochs", 0))
    if warmup_epochs < 0 or full_epochs <= 0:
        raise ValueError("training requires non-negative warmup and positive full-finetune epochs")
    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive")
    if int(training.get("gradient_accumulation_steps", 0)) <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")

    for field in ("train_file", "validation_file", "test_file"):
        if not str(data.get(field, "")).strip():
            raise ValueError(f"data.{field} is required")
    if int(data.get("max_source_length", 0)) <= 0:
        raise ValueError("data.max_source_length must be positive")
    if int(data.get("max_target_length", 0)) <= 1:
        raise ValueError("data.max_target_length must be greater than one")

    checkpoint = config.get("checkpoint", {})
    if bool(checkpoint.get("save_best", False)):
        raise ValueError("LLM2Seq-v2 intentionally supports last.pt only; set checkpoint.save_best=false")
    if bool(checkpoint.get("save_each_epoch", False)):
        raise ValueError("LLM2Seq-v2 intentionally avoids epoch checkpoints")
