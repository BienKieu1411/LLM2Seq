"""Deterministic path resolution for EviSeq-KD inputs and artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def resolve_input_path(value: str, config: Mapping[str, Any]) -> Path:
    """Resolve a configured input, preferring the directory of its config file."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    config_path = config.get("_meta", {}).get("config_path")
    if config_path:
        candidates.append(Path(str(config_path)).resolve().parent / path)

    package_root = Path(__file__).resolve().parents[1]
    if path.parts[:2] == ("src", "eviseq_kd"):
        candidates.append(package_root.joinpath(*path.parts[2:]))
    elif path.parts[:1] == ("eviseq_kd",):
        candidates.append(package_root.joinpath(*path.parts[1:]))

    candidates.extend((Path.cwd() / path, Path(__file__).resolve().parents[3] / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_artifact_path(value: str) -> Path:
    """Resolve a generated artifact relative to the process working directory."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()
