"""Final-only checkpoints with explicit base-weight metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in names}


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = trainable_state_dict(model)
    parameter_names = set(dict(model.named_parameters()))
    stores_full_parameter_state = parameter_names.issubset(state)
    stores_pretrained = any(
        name.startswith(("encoder.model.", "decoder.backbone.", "model.")) for name in state
    )
    torch.save(
        {
            "model_state_dict": state,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "config": config,
            "compact_checkpoint": not stores_pretrained,
            "stores_pretrained_weights": stores_pretrained,
            "stores_full_parameter_state": stores_full_parameter_state,
            "saved_parameter_count": len(parameter_names & set(state)),
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: str | Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    target_names = set(model.state_dict())
    unknown = [name for name in state if name not in target_names]
    if unknown:
        raise RuntimeError(f"Checkpoint tensors do not exist in target model: {unknown[:20]}")
    strict_parameters = bool(
        payload.get(
            "stores_full_parameter_state",
            payload.get("stores_pretrained_weights", False),
        )
    )
    if strict_parameters:
        expected_parameters = set(dict(model.named_parameters()))
        missing_parameters = sorted(expected_parameters - set(state))
        if missing_parameters:
            raise RuntimeError(
                "Full checkpoint is incompatible with the current architecture; "
                "missing parameter tensors: " + ", ".join(missing_parameters[:20])
            )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint tensors: {incompatible.unexpected_keys[:20]}")
    return payload
