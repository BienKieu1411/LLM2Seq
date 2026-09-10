"""Strict configuration loading for the AFMR implementation.

The loader deliberately validates the research contract before any model or
dataset is constructed.  That makes a checkpoint/config mismatch visible at
the command boundary instead of after an expensive run has started.
"""

from __future__ import annotations

import copy
import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as error:  # pragma: no cover - exercised by a useful error path
    raise ImportError("AFMR requires PyYAML; install pyyaml in the active environment") from error


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
    parents = tuple(_parents(own.pop("_base_", None)))
    if len(parents) > 1:
        raise ValueError("AFMR supports only one base config")
    merged: dict[str, Any] = {}
    for parent in parents:
        merged = _merge(merged, _load(path.parent / parent, (*stack, path)))
    return _merge(merged, own)


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    config = _load(resolved, ())
    validate_config(config)
    config.setdefault("_meta", {})["config_path"] = str(resolved)
    config["_meta"]["config_fingerprint"] = config_fingerprint(config)
    return config


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = copy.deepcopy(config)
    serializable.pop("_meta", None)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)


def resolve_path(value: str | Path, config: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_path = config.get("_meta", {}).get("config_path")
    package_root = Path(__file__).resolve().parents[1]
    config_dir = Path(config_path).parent if config_path else package_root
    # Dataset paths in task configs are relative to the repository/package root;
    # paths in an arbitrary external config are relative to that config file.
    base = package_root if config_dir == package_root / "configs" else config_dir
    return (base / path).resolve()


def _check_keys(mapping: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown AFMR {section} key(s): {sorted(unknown)}")


def _finite_positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return number


def _setdefault_training(training: dict[str, Any]) -> None:
    defaults = {
        "interface_warmup_epochs": 1,
        "full_finetune_epochs": 4,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "validation_batch_size": 4,
        "num_workers": 0,
        "validation_num_workers": 0,
        "length_bucketing": True,
        "length_bucket_multiplier": 50,
        "persistent_workers": False,
        "fused_optimizer": False,
        "tf32": False,
        "warmup_bridge_lr": 1e-4,
        "warmup_cross_attention_lr": 1e-4,
        "warmup_grounded_copy_lr": 1e-4,
        "warmup_semantic_read_lr": 1e-4,
        "warmup_dual_readout_lr": 1e-4,
        "full_encoder_lr": 3e-5,
        "full_decoder_lr": 3e-5,
        "full_bridge_lr": 5e-5,
        "full_cross_attention_lr": 5e-5,
        "full_grounded_copy_lr": 5e-5,
        "full_semantic_read_lr": 5e-5,
        "full_dual_readout_lr": 5e-5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 42,
        "log_every_steps": 10,
        "save_each_epoch": True,
        "save_best": True,
        "resume_checkpoint": "",
        "scheduler": "cosine",
        "warmup_steps": 0,
    }
    for key, value in defaults.items():
        training.setdefault(key, value)


def validate_config(config: dict[str, Any]) -> None:
    _check_keys(config, _TOP_LEVEL | {"_meta"}, "top-level")
    required = ("experiment", "model", "encoder", "architecture", "decoder", "training", "data", "generation")
    for section in required:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing AFMR section: {section}")

    _check_keys(config["experiment"], {"name", "output_dir"}, "experiment")
    if not str(config["experiment"].get("output_dir", "")).strip():
        raise ValueError("experiment.output_dir is required")

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
    for key in ("encoder_name", "decoder_name"):
        if not str(model.get(key, "")).strip():
            raise ValueError(f"model.{key} is required")
    if model.get("dtype", "float32") not in {"float32", "bfloat16"}:
        raise ValueError("model.dtype must be float32 or bfloat16")
    if model.get("compute_dtype", "float32") not in {"float32", "bfloat16"}:
        raise ValueError("model.compute_dtype must be float32 or bfloat16")
    if model.get("tokenizer_use_fast", True) is not True:
        raise ValueError("AFMR requires tokenizer_use_fast=true for offset alignment")

    _check_keys(config["encoder"], {"backend", "upper_bidirectional_layers"}, "encoder")
    if config["encoder"].get("backend", "pretrained_native") != "pretrained_native":
        raise ValueError("encoder.backend must be pretrained_native")
    if int(config["encoder"].get("upper_bidirectional_layers", 0)) != 0:
        raise ValueError("upper_bidirectional_layers must be 0 for the native encoder")

    arch = config["architecture"]
    _check_keys(
        arch,
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
    if arch.get("name") != "afmr_value_anchor":
        raise ValueError("architecture.name must be afmr_value_anchor for AFMR")
    for key in ("controller_dim", "depth_rank", "feature_rank", "focus_hidden"):
        if int(arch.get(key, 0)) <= 0:
            raise ValueError(f"architecture.{key} must be positive")
    if int(arch.get("depth_taps", 0)) < 0:
        raise ValueError("architecture.depth_taps must be non-negative")
    windows = tuple(int(width) for width in arch.get("focus_windows", ()))
    if not windows or windows != tuple(sorted(set(windows))) or any(width <= 0 for width in windows):
        raise ValueError("architecture.focus_windows must be strictly increasing positive widths")
    overlap = float(arch.get("focus_overlap", -1.0))
    if not 0.0 <= overlap < 1.0:
        raise ValueError("architecture.focus_overlap must be in [0,1)")
    for width in windows:
        stride = width * (1.0 - overlap)
        if abs(stride - round(stride)) > 1e-8:
            raise ValueError("focus window stride must be an integer")
    for init, maximum in (
        ("depth_gate_init", "depth_gate_max"),
        ("feature_gate_init", "feature_gate_max"),
        ("focus_strength_init", "focus_strength_max"),
    ):
        _finite_positive(arch.get(init, 0), f"architecture.{init}")
        _finite_positive(arch.get(maximum, 0), f"architecture.{maximum}")
        if not float(arch[init]) < float(arch[maximum]) <= 1.0:
            raise ValueError(f"Require 0 < architecture.{init} < {maximum} <= 1")
    tmin, tinit, tmax = (float(arch.get(key, 0)) for key in ("temperature_min", "temperature_init", "temperature_max"))
    if not 0.0 < tmin < tinit < tmax or not all(math.isfinite(value) for value in (tmin, tinit, tmax)):
        raise ValueError("Require finite 0 < temperature_min < temperature_init < temperature_max")

    decoder = config["decoder"]
    _check_keys(
        decoder,
        {
            "cross_attention_every",
            "initialize_cross_from_self",
            "cross_gate_init",
            "cross_gate_max",
            "query_cross_gate",
            "attention_dropout",
            "ce_chunk_size",
            "grounded_copy",
        },
        "decoder",
    )
    if int(decoder.get("cross_attention_every", 0)) != 1:
        raise ValueError("decoder.cross_attention_every must be 1")
    if decoder.get("initialize_cross_from_self", True) is not True:
        raise ValueError("decoder.initialize_cross_from_self must be true")
    if not isinstance(decoder.get("query_cross_gate", False), bool):
        raise ValueError("decoder.query_cross_gate must be boolean")
    cross_gate_init = _finite_positive(decoder.get("cross_gate_init", 0.10), "decoder.cross_gate_init")
    cross_gate_max = _finite_positive(decoder.get("cross_gate_max", 1.0), "decoder.cross_gate_max")
    if not cross_gate_init < cross_gate_max <= 1.0:
        raise ValueError("Require 0 < decoder.cross_gate_init < cross_gate_max <= 1")
    _finite_positive(decoder.get("attention_dropout", 0.0), "decoder.attention_dropout", allow_zero=True)
    if float(decoder.get("attention_dropout", 0.0)) >= 1.0:
        raise ValueError("decoder.attention_dropout must be below 1")
    if int(decoder.get("ce_chunk_size", 0)) <= 0:
        raise ValueError("decoder.ce_chunk_size must be positive")
    copy_cfg = decoder.get("grounded_copy")
    if not isinstance(copy_cfg, dict):
        raise ValueError("decoder.grounded_copy must be a mapping")
    _check_keys(
        copy_cfg,
        {
            "enabled",
            "key_dim",
            "gate_init",
            "readout_mode",
            "alpha_max",
            "generate_reserve",
            "alpha_init",
            "base_floor",
            "hidden_lambda",
            "detach_copy_route_features",
            "copy_entropy_feature",
            "hard_source_fallback",
            "logit_offset",
            "null_slot",
            "semantic_read",
        },
        "decoder.grounded_copy",
    )
    if not isinstance(copy_cfg.get("enabled", False), bool):
        raise ValueError("grounded_copy.enabled must be boolean")
    if int(copy_cfg.get("key_dim", 0)) <= 0:
        raise ValueError("grounded_copy.key_dim must be positive")
    _finite_positive(copy_cfg.get("gate_init", 0.05), "grounded_copy.gate_init")
    if float(copy_cfg.get("gate_init", 0.05)) >= 1.0:
        raise ValueError("grounded_copy.gate_init must be below 1")
    modes = {
        "copy_mass_preserving_capped_simplex",
        "independent_capped_simplex",
        "legacy_copy_mixture",
        "hidden_interpolation",
        "constant_alpha",
    }
    if copy_cfg.get("readout_mode") not in modes:
        raise ValueError(f"grounded_copy.readout_mode must be one of {sorted(modes)}")
    alpha_max = _finite_positive(copy_cfg.get("alpha_max", 0.20), "grounded_copy.alpha_max")
    reserve = _finite_positive(
        copy_cfg.get("generate_reserve", 0.05), "grounded_copy.generate_reserve", allow_zero=True
    )
    alpha_init = _finite_positive(copy_cfg.get("alpha_init", 0.05), "grounded_copy.alpha_init")
    floor = _finite_positive(copy_cfg.get("base_floor", 0.05), "grounded_copy.base_floor", allow_zero=True)
    if alpha_max > 1.0 or reserve >= 1.0 or floor >= 1.0 or alpha_init >= alpha_max:
        raise ValueError("Invalid grounded-copy simplex bounds")
    _finite_positive(copy_cfg.get("hidden_lambda", 0.10), "grounded_copy.hidden_lambda", allow_zero=True)
    control_defaults = {"detach_copy_route_features": True, "hard_source_fallback": True, "null_slot": False}
    for key, default in control_defaults.items():
        if not isinstance(copy_cfg.get(key, default), bool):
            raise ValueError(f"grounded_copy.{key} must be boolean")
    if copy_cfg.get("copy_entropy_feature", "normalized_attention_entropy") != "normalized_attention_entropy":
        raise ValueError("AFMR supports only copy_entropy_feature=normalized_attention_entropy")
    if copy_cfg.get("null_slot", False):
        raise ValueError("grounded_copy.null_slot is reserved for a later diagnostic variant")
    if copy_cfg.get("logit_offset", "z0_log_partition") != "z0_log_partition":
        raise ValueError("AFMR uses grounded_copy.logit_offset=z0_log_partition")

    semantic = copy_cfg.get("semantic_read")
    if not isinstance(semantic, dict):
        raise ValueError("decoder.grounded_copy.semantic_read must be a mapping")
    _check_keys(
        semantic,
        {
            "enabled",
            "rank",
            "num_heads",
            "key_source",
            "value_source",
            "semantic_prior_scale",
            "max_relative_rms",
            "cap_mode",
            "inner_gate",
            "gate_init",
            "output_init",
        },
        "decoder.grounded_copy.semantic_read",
    )
    if not isinstance(semantic.get("enabled", False), bool):
        raise ValueError("semantic_read.enabled must be boolean")
    if int(semantic.get("rank", 0)) <= 0 or int(semantic.get("num_heads", 1)) != 1:
        raise ValueError("AFMR main uses one semantic head with positive rank")
    if semantic.get("key_source") not in {"H0", "M"} or semantic.get("value_source") != "H0":
        raise ValueError("semantic_read must use key_source H0/M and value_source H0")
    _finite_positive(semantic.get("semantic_prior_scale", 1.0), "semantic_read.semantic_prior_scale", allow_zero=True)
    _finite_positive(semantic.get("max_relative_rms", 0.10), "semantic_read.max_relative_rms")
    if semantic.get("cap_mode", "smooth_relative_rms") not in {"smooth_relative_rms", "legacy"}:
        raise ValueError("semantic_read.cap_mode must be smooth_relative_rms or legacy")
    if not isinstance(semantic.get("inner_gate", False), bool):
        raise ValueError("semantic_read.inner_gate must be boolean")
    _finite_positive(semantic.get("gate_init", 0.05), "semantic_read.gate_init")
    if float(semantic.get("gate_init", 0.05)) >= 1.0:
        raise ValueError("semantic_read.gate_init must be below 1")
    if semantic.get("output_init") not in {"zero", "tiny_rms_1e-3"}:
        raise ValueError("semantic_read.output_init must be zero or tiny_rms_1e-3")
    if copy_cfg.get("readout_mode") == "legacy_copy_mixture":
        if not copy_cfg.get("enabled", False):
            raise ValueError("legacy_copy_mixture readout requires grounded_copy.enabled=true")
        if not semantic.get("enabled", False):
            raise ValueError("legacy_copy_mixture readout requires semantic_read.enabled=true")
        if not semantic.get("inner_gate", False) or semantic.get("output_init") != "zero":
            raise ValueError("legacy_copy_mixture readout requires inner_gate=true and output_init=zero")
        if semantic.get("cap_mode") != "legacy":
            raise ValueError("legacy_copy_mixture readout requires semantic_read.cap_mode=legacy")
    if copy_cfg.get("readout_mode") == "hidden_interpolation" and not semantic.get("enabled", False):
        raise ValueError("hidden_interpolation requires semantic_read.enabled=true")

    training = config["training"]
    _setdefault_training(training)
    allowed_training = {
        "interface_warmup_epochs",
        "full_finetune_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "validation_batch_size",
        "num_workers",
        "validation_num_workers",
        "warmup_bridge_lr",
        "warmup_cross_attention_lr",
        "warmup_grounded_copy_lr",
        "warmup_semantic_read_lr",
        "warmup_dual_readout_lr",
        "full_encoder_lr",
        "full_decoder_lr",
        "full_bridge_lr",
        "full_cross_attention_lr",
        "full_grounded_copy_lr",
        "full_semantic_read_lr",
        "full_dual_readout_lr",
        "weight_decay",
        "max_grad_norm",
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
        "scheduler",
        "warmup_steps",
    }
    _check_keys(training, allowed_training, "training")
    for key in ("interface_warmup_epochs", "full_finetune_epochs"):
        if int(training[key]) < 0:
            raise ValueError(f"training.{key} must be non-negative")
    if int(training["interface_warmup_epochs"]) + int(training["full_finetune_epochs"]) <= 0:
        raise ValueError("At least one training epoch is required")
    for key in ("batch_size", "gradient_accumulation_steps", "validation_batch_size", "log_every_steps"):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(training["length_bucket_multiplier"]) <= 0 or int(training["warmup_steps"]) < 0:
        raise ValueError("training.length_bucket_multiplier must be positive and warmup_steps non-negative")
    for key, value in training.items():
        if key.endswith("_lr") or key in {"weight_decay"}:
            _finite_positive(value, f"training.{key}", allow_zero=key == "weight_decay")
    max_grad_norm = training.get("max_grad_norm")
    if max_grad_norm is not None:
        _finite_positive(max_grad_norm, "training.max_grad_norm")
    if training.get("scheduler") != "cosine":
        raise ValueError("training.scheduler must be cosine")

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
            "decoder_chat_template",
            "decoder_prefix",
            "detokenize",
            "max_source_length",
            "max_target_length",
        },
        "data",
    )
    for key in ("train_file", "validation_file", "test_file", "source_field", "target_field"):
        if not str(data.get(key, "")).strip():
            raise ValueError(f"data.{key} is required")
    for key in ("max_source_length", "max_target_length"):
        if int(data.get(key, 0)) <= 0:
            raise ValueError(f"data.{key} must be positive")

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
            "compact_finished",
            "temperature",
            "top_p",
        },
        "generation",
    )
    if int(generation.get("batch_size", 0)) <= 0 or int(generation.get("max_new_tokens", 0)) <= 0:
        raise ValueError("generation.batch_size and max_new_tokens must be positive")
    if not 0 <= int(generation.get("min_new_tokens", 0)) < int(generation.get("max_new_tokens", 0)):
        raise ValueError("Require 0 <= generation.min_new_tokens < max_new_tokens")
    if int(generation.get("num_beams", 1)) != 1:
        raise ValueError("Benchmark generation uses num_beams=1")
    if not isinstance(generation.get("do_sample", False), bool):
        raise ValueError("generation.do_sample must be boolean")
    _finite_positive(generation.get("repetition_penalty", 1.0), "generation.repetition_penalty")
    if int(generation.get("no_repeat_ngram_size", 0)) < 0:
        raise ValueError("generation.no_repeat_ngram_size must be non-negative")
    temperature = _finite_positive(generation.get("temperature", 1.0), "generation.temperature")
    top_p = _finite_positive(generation.get("top_p", 1.0), "generation.top_p")
    if top_p > 1.0 or temperature <= 0.0:
        raise ValueError("generation requires 0 < temperature and 0 < top_p <= 1")


def config_fingerprint(config: dict[str, Any]) -> str:
    """Hash the resolved research configuration, excluding runtime metadata."""

    payload = copy.deepcopy(config)
    payload.pop("_meta", None)
    # Output locations and a resume source are run state, not model/data
    # semantics.  Including either would make a checkpoint saved with
    # ``--output-dir`` impossible to evaluate from the same YAML recipe.
    payload.get("experiment", {}).pop("output_dir", None)
    payload.get("training", {}).pop("resume_checkpoint", None)
    for key in ("train_file", "validation_file", "test_file"):
        if key in payload.get("data", {}):
            payload["data"][key] = str(resolve_path(payload["data"][key], config))
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


__all__ = ["config_fingerprint", "dump_config", "load_config", "resolve_path", "validate_config"]
