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
    data_manifest: Dict[str, Any],
) -> None:
    path = Path(path)
    if path.name != "last.pt":
        raise ValueError("LLM2Seq-v2 only writes the canonical last.pt checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    expected = set(model.state_dict())
    if set(state) != expected:
        raise RuntimeError("Checkpoint serialization omitted model tensors")
    parameter_names = set(dict(model.named_parameters()))
    components: Dict[str, Dict[str, int]] = {}
    for name, value in state.items():
        component = name.split(".", maxsplit=1)[0]
        entry = components.setdefault(component, {"tensors": 0, "elements": 0})
        entry["tensors"] += 1
        entry["elements"] += int(value.numel())
    payload = {
        "model_state_dict": state,
        "checkpoint_role": "last",
        "stores_full_model_state": True,
        "stores_full_parameter_state": parameter_names.issubset(state),
        "stores_pretrained_weights": True,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
        "data_manifest": data_manifest,
        "component_manifest": components,
        "saved_tensor_count": len(state),
        "saved_parameter_count": len(parameter_names),
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
        raise RuntimeError(
            f"Incompatible full checkpoint; missing={missing[:10]}, unknown={unknown[:10]}"
        )
    model.load_state_dict(state, strict=True)
    return payload
