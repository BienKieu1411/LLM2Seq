"""One complete, atomic last.pt checkpoint; no best/epoch checkpoint paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


def save_last_checkpoint(
    model: nn.Module,
    path: str | Path,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
) -> None:
    path = Path(path)
    if path.name != "last.pt":
        raise ValueError("EviSeq only writes the canonical last.pt checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    payload = {
        "model_state_dict": state,
        "checkpoint_role": "last",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_last_checkpoint(model: nn.Module, path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_role") != "last":
        raise RuntimeError(f"Expected a last checkpoint, found {payload.get('checkpoint_role')!r}")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint has no model_state_dict")
    expected = set(model.state_dict())
    present = set(state)
    if expected != present:
        missing = sorted(expected - present)
        unknown = sorted(present - expected)
        raise RuntimeError(f"Incompatible full checkpoint; missing={missing[:10]}, unknown={unknown[:10]}")
    model.load_state_dict(state, strict=True)
    return payload


def initialize_from_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """Initialize a new task run from an EviSeq checkpoint.

    Strict mode requires the same model graph. Partial mode loads every tensor
    whose name and shape match, which is useful when task-specific training
    heads are enabled or disabled between runs.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a model state dictionary")
    if strict:
        model.load_state_dict(state, strict=True)
        loaded = len(state)
        skipped: list[str] = []
    else:
        expected = model.state_dict()
        compatible = {
            name: value
            for name, value in state.items()
            if name in expected and tuple(value.shape) == tuple(expected[name].shape)
        }
        if not compatible:
            raise RuntimeError("No compatible parameters were found in the initialization checkpoint")
        model.load_state_dict(compatible, strict=False)
        loaded = len(compatible)
        skipped = sorted(set(state) - set(compatible))
    return {
        "epoch": int(payload.get("epoch", 0)) if isinstance(payload, dict) else 0,
        "global_step": int(payload.get("global_step", 0)) if isinstance(payload, dict) else 0,
        "loaded_tensors": loaded,
        "skipped_tensors": skipped,
    }
