"""Best/last checkpoints with explicit base-weight metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _full_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Copy every parameter and persistent buffer to CPU.

    Saving only currently trainable tensors is unsafe for a best checkpoint:
    the best epoch can occur during interface warm-up, when both pretrained
    Qwen backbones are frozen. Best and last are therefore always complete,
    self-contained model states independent of ``requires_grad``.
    """

    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def _component_manifest(state: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, int]]:
    manifest: Dict[str, Dict[str, int]] = {}
    for name, tensor in state.items():
        component = name.split(".", maxsplit=1)[0]
        entry = manifest.setdefault(component, {"tensors": 0, "elements": 0})
        entry["tensors"] += 1
        entry["elements"] += int(tensor.numel())
    return manifest


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
    validation_metrics: Optional[Dict[str, float]] = None,
    checkpoint_role: Optional[str] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _full_state_dict(model)
    expected_state_names = set(model.state_dict())
    if set(state) != expected_state_names:  # pragma: no cover - defensive
        missing = sorted(expected_state_names - set(state))
        raise RuntimeError("Checkpoint serialization omitted model tensors: " + ", ".join(missing[:20]))
    parameter_names = set(dict(model.named_parameters()))
    stores_full_parameter_state = parameter_names.issubset(state)
    stores_pretrained = any(name.startswith(("encoder.model.", "decoder.backbone.", "model.")) for name in state)
    payload = {
        "model_state_dict": state,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
        "compact_checkpoint": False,
        "stores_pretrained_weights": stores_pretrained,
        "stores_full_parameter_state": stores_full_parameter_state,
        "stores_full_model_state": True,
        "saved_tensor_count": len(state),
        "saved_parameter_count": len(parameter_names & set(state)),
        "component_manifest": _component_manifest(state),
    }
    if validation_metrics is not None:
        payload["validation_metrics"] = {str(name): float(value) for name, value in validation_metrics.items()}
    if checkpoint_role is not None:
        if checkpoint_role not in {"best", "last"}:
            raise ValueError("checkpoint_role must be 'best' or 'last'")
        payload["checkpoint_role"] = checkpoint_role

    # Never leave a half-written multi-GB checkpoint under the canonical name
    # when a run is interrupted during serialization.
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(model: nn.Module, path: str | Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    target_names = set(model.state_dict())
    unknown = [name for name in state if name not in target_names]
    if unknown:
        raise RuntimeError(f"Checkpoint tensors do not exist in target model: {unknown[:20]}")
    strict_model_state = bool(payload.get("stores_full_model_state", False))
    strict_parameters = bool(
        payload.get(
            "stores_full_parameter_state",
            payload.get("stores_pretrained_weights", False),
        )
    )
    if strict_model_state:
        missing_state = sorted(target_names - set(state))
        if missing_state:
            raise RuntimeError(
                "Full checkpoint is incompatible with the current architecture; "
                "missing model tensors: " + ", ".join(missing_state[:20])
            )
    elif strict_parameters:
        expected_parameters = set(dict(model.named_parameters()))
        missing_parameters = sorted(expected_parameters - set(state))
        if missing_parameters:
            raise RuntimeError(
                "Full checkpoint is incompatible with the current architecture; "
                "missing parameter tensors: " + ", ".join(missing_parameters[:20])
            )
    incompatible = model.load_state_dict(state, strict=strict_model_state)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint tensors: {incompatible.unexpected_keys[:20]}")
    return payload
