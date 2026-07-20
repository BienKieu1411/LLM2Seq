"""YAML configuration loading with small, explicit base-file inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config.

    An optional ``_base_`` entry is resolved relative to the child config. Base
    inheritance is intentionally limited to one path per file so every
    experiment remains easy to audit.
    """

    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    if not isinstance(current, dict):
        raise TypeError(f"Config root must be a mapping: {path}")

    base_ref = current.pop("_base_", None)
    if base_ref is None:
        merged = current
    else:
        base_path = (path.parent / str(base_ref)).resolve()
        merged = _deep_merge(load_config(base_path), current)

    merged.setdefault("_meta", {})["config_path"] = str(path)
    return merged


def dump_config(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
