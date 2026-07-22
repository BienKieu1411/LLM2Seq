"""YAML configuration loading and controlled model-size overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

MODEL_PROFILES = {
    "0.6B": {
        "name": "Qwen/Qwen3-0.6B",
        "layers": 28,
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw_torch",
        "full_lr": 1.0e-5,
    },
    "1.7B": {
        "name": "Qwen/Qwen3-1.7B",
        "layers": 28,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "optimizer": "adamw_torch",
        "full_lr": 8.0e-6,
    },
    "4B": {
        "name": "Qwen/Qwen3-4B",
        "layers": 36,
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "optimizer": "adamw_8bit",
        "full_lr": 5.0e-6,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    if not isinstance(current, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    base_ref = current.pop("_base_", None)
    merged = current if base_ref is None else _deep_merge(
        load_config((path.parent / str(base_ref)).resolve()), current
    )
    merged.setdefault("_meta", {})["config_path"] = str(path)
    return merged


def dump_config(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def apply_model_size(config: Dict[str, Any], model_size: str | None) -> Dict[str, Any]:
    if model_size is None:
        return config
    if model_size not in MODEL_PROFILES:
        raise ValueError(f"model_size must be one of {sorted(MODEL_PROFILES)}")
    profile = MODEL_PROFILES[model_size]
    name = str(profile["name"])
    previous = str((config.get("model", {}) or {}).get("encoder_name", ""))
    config.setdefault("model", {})["encoder_name"] = name
    config.setdefault("decoder", {})["pretrained_name"] = name
    config.setdefault("decoder", {})["num_layers"] = int(profile["layers"])
    # These profiles all keep an effective batch of 32. Explicit values in a
    # dedicated model config can still override them after loading.
    config.setdefault("training", {})["batch_size"] = int(profile["batch_size"])
    config.setdefault("training", {})["gradient_accumulation_steps"] = int(
        profile["gradient_accumulation_steps"]
    )
    config.setdefault("training", {})["optimizer"] = str(profile["optimizer"])
    config.setdefault("training", {})["full_lr"] = float(profile["full_lr"])
    if previous and previous != name:
        experiment = config.setdefault("experiment", {})
        output = Path(str(experiment.get("output_dir", "runs/genbridge")))
        suffix = model_size.lower().replace(".", "_")
        experiment["output_dir"] = str(output.with_name(f"{output.name}_{suffix}"))
    return config
