"""Strict source-film configuration loading and invariant checks."""

from __future__ import annotations

import copy
import math
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


def contextual_value_settings(architecture: dict[str, Any]) -> dict[str, Any]:
    """Compatibility shim: source-film never uses a contextual value branch."""
    return {"enabled": False, "max_relative_rms": 0.0}


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
            "compute_dtype",
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
    if model.get("compute_dtype", "bfloat16") not in {"float32", "bfloat16"}:
        raise ValueError("model.compute_dtype must be float32 or bfloat16")
    if not model.get("tokenizer_use_fast", True):
        raise ValueError("AFMR requires a fast encoder tokenizer for exact offset mapping")
    architecture = config["architecture"]
    _check_keys(
        architecture,
        {
            "name",
            "bridge_mode",
            "controller_dim",
            "film_rank",
            "film_gate_init",
            "film_gate_max",
            "film_max_relative_rms",
        },
        "architecture",
    )
    if architecture.get("name") != "source_conditioned_adaptive_norm":
        raise ValueError("architecture.name must be source_conditioned_adaptive_norm")
    if architecture.get("bridge_mode", "source_film") not in {"source_film", "direct_projection"}:
        raise ValueError("architecture.bridge_mode must be source_film or direct_projection")
    contextual_value_settings(architecture)
    for key in ("controller_dim", "film_rank"):
        if int(architecture.get(key, 0)) <= 0:
            raise ValueError(f"architecture.{key} must be positive")
    if not 0 < float(architecture.get("film_gate_init", 0.05)) < float(architecture.get("film_gate_max", 0.20)) <= 1:
        raise ValueError("Require 0 < architecture.film_gate_init < film_gate_max <= 1")
    if not 0 < float(architecture.get("film_max_relative_rms", 0.10)) <= 1:
        raise ValueError("architecture.film_max_relative_rms must lie in (0, 1]")
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
            "ce_chunk_size",
            "grounded_copy",
            "source_film",
        },
        "decoder",
    )
    if int(decoder.get("cross_attention_every", 0)) != 1:
        raise ValueError("AFMR uses cross-attention in every decoder layer")
    if not bool(decoder.get("initialize_cross_from_self", True)):
        raise ValueError("AFMR cross-attention projections must be initialized from decoder self-attention")
    copy_config = decoder.get("grounded_copy", {})
    if not isinstance(copy_config, dict):
        raise ValueError("decoder.grounded_copy must be a mapping")
    _check_keys(copy_config, {"enabled", "key_dim", "gate_init"}, "decoder.grounded_copy")
    if not isinstance(copy_config.get("enabled", False), bool):
        raise ValueError("decoder.grounded_copy.enabled must be a boolean")
    if int(copy_config.get("key_dim", 128)) <= 0 or not 0 < float(copy_config.get("gate_init", 0.05)) < 1:
        raise ValueError("Grounded copy requires key_dim > 0 and 0 < gate_init < 1")
    film_config = decoder.get("source_film", {})
    if not isinstance(film_config, dict):
        raise ValueError("decoder.source_film must be a mapping")
    _check_keys(film_config, {"enabled", "rank", "gate_init", "gate_max", "max_relative_rms"}, "decoder.source_film")
    if not isinstance(film_config.get("enabled", True), bool):
        raise ValueError("decoder.source_film.enabled must be a boolean")
    if int(film_config.get("rank", 128)) <= 0:
        raise ValueError("decoder.source_film.rank must be positive")
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
            "salience_loss_weight",
            "salience_margin",
            "seed",
            "log_every_steps",
            "save_each_epoch",
            "save_best",
            "resume_checkpoint",
            "length_bucketing",
            "length_bucket_multiplier",
            "persistent_workers",
            "fused_optimizer",
            "tf32",
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
    salience_weight = float(training.get("salience_loss_weight", 0.0))
    salience_margin = float(training.get("salience_margin", 0.5))
    if not math.isfinite(salience_weight) or salience_weight < 0:
        raise ValueError("training.salience_loss_weight must be non-negative")
    if not math.isfinite(salience_margin) or salience_margin < 0:
        raise ValueError("training.salience_margin must be non-negative")
    if salience_weight != 0:
        raise ValueError("source-film training is CE-only; training.salience_loss_weight must be 0")
    data = config["data"]
    if int(decoder.get("ce_chunk_size", 1024)) <= 0:
        raise ValueError("decoder.ce_chunk_size must be positive")
    if int(training.get("length_bucket_multiplier", 50)) <= 0:
        raise ValueError("training.length_bucket_multiplier must be positive")
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
            "system_prompt",
            "system_prompt_field",
            "decoder_prompt",
            "decoder_chat_template",
            "decoder_prefix",
            "detokenize",
            "max_source_length",
            "max_target_length",
        },
        "data",
    )
    for name in ("train_file", "validation_file", "test_file", "source_field", "target_field"):
        if not str(data.get(name, "")).strip():
            raise ValueError(f"data.{name} is required")
    for name in ("system_prompt", "system_prompt_field"):
        value = data.get(name, "")
        if value is not None and not isinstance(value, str):
            raise ValueError(f"data.{name} must be a string when provided")
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
            "temperature",
            "top_k",
            "top_p",
            "compact_finished",
        },
        "generation",
    )
    if int(generation.get("num_beams", 0)) != 1:
        raise ValueError("AFMR generation currently supports num_beams=1 only")
    if float(generation.get("repetition_penalty", 1.0)) <= 0:
        raise ValueError("generation.repetition_penalty must be positive")
    if int(generation.get("no_repeat_ngram_size", 0)) < 0:
        raise ValueError("generation.no_repeat_ngram_size must be non-negative")
    temperature = float(generation.get("temperature", 0.0))
    if bool(generation.get("do_sample", False)) and temperature <= 0:
        raise ValueError("generation.temperature must be positive when do_sample=true")
    if not bool(generation.get("do_sample", False)) and temperature < 0:
        raise ValueError("generation.temperature must be non-negative")
    if int(generation.get("top_k", 0)) < 0:
        raise ValueError("generation.top_k must be non-negative")
    if not 0 < float(generation.get("top_p", 1.0)) <= 1:
        raise ValueError("generation.top_p must lie in (0, 1]")
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
