from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    required_sections = ("target", "data", "generation", "draft", "source_memory", "speculation")
    missing = [section for section in required_sections if not isinstance(config.get(section), dict)]
    if missing:
        raise ValueError(f"Missing YAML sections: {', '.join(missing)}")

    if not bool(config["target"].get("local_files_only", True)):
        raise ValueError("Remote model loading is disabled; set target.local_files_only: true")

    generation = config["generation"]
    if int(generation.get("num_beams", 1)) != 1:
        raise ValueError("This implementation supports greedy single-beam generation only")
    if float(generation.get("temperature", 0.0)) != 0.0:
        raise ValueError("Sampling is not implemented; set generation.temperature: 0.0")
    if int(generation.get("top_k", 0)) != 0:
        raise ValueError("Sampling is not implemented; set generation.top_k: 0")
    if float(generation.get("top_p", 1.0)) != 1.0:
        raise ValueError("Sampling is not implemented; set generation.top_p: 1.0")
    if int(generation.get("max_new_tokens", 0)) < 0:
        raise ValueError("generation.max_new_tokens must be nonnegative")
    if int(generation.get("min_new_tokens", 0)) < 0:
        raise ValueError("generation.min_new_tokens must be nonnegative")

    draft = config["draft"]
    width = int(draft.get("width", 0))
    heads = int(draft.get("heads", 0))
    if width < 1 or heads < 1 or width % heads:
        raise ValueError("draft.width must be positive and divisible by draft.heads")
    for key in ("layers", "ffn_width", "max_block_size", "target_feature_window", "max_source_tokens"):
        if int(draft.get(key, 0)) < 1:
            raise ValueError(f"draft.{key} must be positive")

    mode = str(config["source_memory"].get("mode", "full"))
    if mode not in {"full", "ordered_pool"}:
        raise ValueError("source_memory.mode must be 'full' or 'ordered_pool'")
    if mode == "ordered_pool" and int(config["source_memory"].get("budget", 0)) < 1:
        raise ValueError("source_memory.budget must be positive for ordered_pool")

    block_size = int(config["speculation"].get("block_size", 0))
    if block_size < 0 or block_size > int(draft["max_block_size"]):
        raise ValueError("speculation.block_size must lie between 0 and draft.max_block_size")


def resolve_path(value: str | Path, config: Dict[str, Any]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config["_config_dir"]) / path).resolve()


def torch_dtype(name: str):
    import torch

    normalized = str(name).lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported target.dtype: {name}") from exc
