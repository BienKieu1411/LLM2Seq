"""Strict AFMR configuration loading and invariant checks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

_TOP_LEVEL = {
    "experiment",
    "model",
    "encoder",
    "architecture",
    "decoder",
    "training",
    "data",
    "generation",
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parents(value: Any) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise TypeError("_base_ must be a path or list of paths")


def _load(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        raise ValueError("Cyclic config inheritance: " + " -> ".join(map(str, (*stack, path))))
    if len(stack) > 1:
        raise ValueError("AFMR supports only one base and one task override")
    with path.open("r", encoding="utf-8") as handle:
        own = yaml.safe_load(handle) or {}
    if not isinstance(own, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    merged: dict[str, Any] = {}
    parents = tuple(_parents(own.pop("_base_", None)))
    if len(parents) > 1:
        raise ValueError("AFMR supports only one base config")
    for parent in parents:
        merged = _merge(merged, _load(path.parent / parent, (*stack, path)))
    return _merge(merged, own)


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    config = _load(resolved, ())
    validate_config(config)
    config.setdefault("_meta", {})["config_path"] = str(resolved)
    return config


def resolve_path(value: str | Path, config: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_path = config.get("_meta", {}).get("config_path")
    package_root = Path(__file__).resolve().parents[1]
    config_dir = Path(config_path).parent if config_path else package_root
    base = package_root if config_dir == package_root / "configs" else config_dir
    return (base / path).resolve()


def _check_keys(mapping: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown AFMR {section} key(s): {sorted(unknown)}")


def validate_config(config: dict[str, Any]) -> None:
    _check_keys(config, _TOP_LEVEL | {"_meta"}, "top-level")
    required_sections = ("model", "encoder", "architecture", "decoder", "training", "data", "generation")
    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing AFMR section: {section}")
    model = config["model"]
    _check_keys(
        model,
        {
            "encoder_name",
            "decoder_name",
            "dtype",
            "tokenizer_use_fast",
            "gradient_checkpointing",
            "attention_implementation",
            "trust_remote_code",
        },
        "model",
    )
    if not str(model.get("encoder_name", "")).strip() or not str(model.get("decoder_name", "")).strip():
        raise ValueError("model.encoder_name and model.decoder_name are required")
    if model.get("dtype", "float32") not in {"float32", "bfloat16"}:
        raise ValueError("AFMR supports float32 or bfloat16; float16 requires a loss scaler and is not supported")
    if not model.get("tokenizer_use_fast", True):
        raise ValueError("AFMR requires a fast encoder tokenizer for exact offset mapping")
    architecture = config["architecture"]
    _check_keys(
        architecture,
        {
            "name",
            "controller_dim",
            "depth_taps",
            "depth_rank",
            "depth_gate_init",
            "depth_gate_max",
            "feature_rank",
            "feature_gate_init",
            "feature_gate_max",
            "focus_hidden",
            "focus_windows",
            "focus_overlap",
            "focus_strength_init",
            "focus_strength_max",
            "temperature_init",
            "temperature_min",
            "temperature_max",
        },
        "architecture",
    )
    if architecture.get("name") != "afmr_v1":
        raise ValueError("architecture.name must be afmr_v1")
    taps = int(architecture.get("depth_taps", 0))
    if taps < 0:
        raise ValueError("architecture.depth_taps must be non-negative")
    windows = architecture.get("focus_windows", [])
    if not windows or any(int(w) <= 0 for w in windows):
        raise ValueError("architecture.focus_windows must contain positive widths")
    normalized_windows = tuple(int(width) for width in windows)
    if normalized_windows != tuple(sorted(set(normalized_windows))):
        raise ValueError("architecture.focus_windows must be strictly increasing")
    overlap = float(architecture.get("focus_overlap", -1))
    if not 0.0 <= overlap < 1.0:
        raise ValueError("architecture.focus_overlap must be in [0, 1)")
    if any(abs(w * (1 - overlap) - round(w * (1 - overlap))) > 1e-8 for w in normalized_windows):
        raise ValueError("focus window stride must be an integer")
    for key in ("controller_dim", "depth_rank", "feature_rank", "focus_hidden"):
        if int(architecture.get(key, 0)) <= 0:
            raise ValueError(f"architecture.{key} must be positive")
    for name in ("depth_gate_init", "feature_gate_init", "focus_strength_init"):
        if float(architecture.get(name, -1)) < 0:
            raise ValueError(f"architecture.{name} must be non-negative")
    for init, maximum in (
        ("depth_gate_init", "depth_gate_max"),
        ("feature_gate_init", "feature_gate_max"),
        ("focus_strength_init", "focus_strength_max"),
    ):
        if not 0 < float(architecture[init]) < float(architecture[maximum]):
            raise ValueError(f"Require 0 < architecture.{init} < {maximum}")
    if float(architecture.get("temperature_min", 0)) <= 0 or float(architecture.get("temperature_max", 0)) < float(
        architecture.get("temperature_min", 0)
    ):
        raise ValueError("Invalid focus temperature bounds")
    if (
        not float(architecture["temperature_min"])
        < float(architecture.get("temperature_init", 0))
        < float(architecture["temperature_max"])
    ):
        raise ValueError("architecture.temperature_init must lie strictly within temperature bounds")
    _check_keys(config["encoder"], {"backend", "upper_bidirectional_layers"}, "encoder")
    if config["encoder"].get("backend", "pretrained_native") != "pretrained_native":
        raise ValueError("AFMR currently exposes only encoder.backend=pretrained_native")
    if int(config["encoder"].get("upper_bidirectional_layers", 0)) != 0:
        raise ValueError(
            "AFMR pretrained_native keeps the encoder attention implementation unchanged; upper_bidirectional_layers must be 0"
        )
    decoder = config["decoder"]
    _check_keys(
        decoder,
        {
            "cross_attention_every",
            "initialize_cross_from_self",
            "cross_gate_init",
            "cross_gate_max",
            "attention_dropout",
        },
        "decoder",
    )
    if int(decoder.get("cross_attention_every", 0)) != 1:
        raise ValueError("AFMR uses cross-attention in every decoder layer")
    if not bool(decoder.get("initialize_cross_from_self", True)):
        raise ValueError("AFMR cross-attention projections must be initialized from decoder self-attention")
    training = config["training"]
    _check_keys(
        training,
        {
            "interface_warmup_epochs",
            "full_finetune_epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "validation_batch_size",
            "num_workers",
            "validation_num_workers",
            "warmup_bridge_lr",
            "warmup_cross_attention_lr",
            "full_encoder_lr",
            "full_decoder_lr",
            "full_bridge_lr",
            "full_cross_attention_lr",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "log_every_steps",
            "save_each_epoch",
            "save_best",
            "resume_checkpoint",
        },
        "training",
    )
    for name in ("interface_warmup_epochs", "full_finetune_epochs", "batch_size", "gradient_accumulation_steps"):
        if int(training.get(name, 0)) < 0:
            raise ValueError(f"training.{name} must be non-negative")
    if int(training.get("batch_size", 0)) == 0 or int(training.get("gradient_accumulation_steps", 0)) == 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    if int(training.get("interface_warmup_epochs", 0)) + int(training.get("full_finetune_epochs", 0)) == 0:
        raise ValueError("At least one AFMR training epoch is required")
    data = config["data"]
    _check_keys(
        data,
        {
            "train_file",
            "validation_file",
            "test_file",
            "source_field",
            "target_field",
            "id_field",
            "list_separator",
            "encoder_prefix",
            "decoder_prompt",
            "max_source_length",
            "max_target_length",
        },
        "data",
    )
    for name in ("train_file", "validation_file", "test_file", "source_field", "target_field"):
        if not str(data.get(name, "")).strip():
            raise ValueError(f"data.{name} is required")
    generation = config["generation"]
    _check_keys(
        generation,
        {
            "batch_size",
            "max_new_tokens",
            "min_new_tokens",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "num_beams",
            "do_sample",
        },
        "generation",
    )
    if int(generation.get("num_beams", 0)) != 1 or bool(generation.get("do_sample", True)):
        raise ValueError("AFMR evaluation is greedy: num_beams=1 and do_sample=false")
    if float(generation.get("repetition_penalty", 1.0)) != 1 or int(generation.get("no_repeat_ngram_size", 0)) != 0:
        raise ValueError("AFMR greedy currently supports repetition_penalty=1 and no_repeat_ngram_size=0 only")
    if not 0 <= int(generation.get("min_new_tokens", 0)) < int(generation.get("max_new_tokens", 0)):
        raise ValueError("Require 0 <= min_new_tokens < max_new_tokens")
    for section, key in (
        ("generation", "batch_size"),
        ("training", "validation_batch_size"),
        ("training", "log_every_steps"),
        ("data", "max_source_length"),
        ("data", "max_target_length"),
    ):
        if int(config[section].get(key, 0)) <= 0:
            raise ValueError(f"{section}.{key} must be positive")
