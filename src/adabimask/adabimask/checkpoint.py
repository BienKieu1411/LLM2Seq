"""Compact checkpoints that never duplicate frozen base-model weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    full_state = model.state_dict()
    return {name: tensor.detach().cpu() for name, tensor in full_state.items() if name in trainable_names}


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
    eval_loss: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = trainable_state_dict(model)
    payload: Dict[str, Any] = {
        "model_state_dict": state,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "eval_loss": float(eval_loss),
        "config": config,
        "compact_checkpoint": True,
        "stores_base_encoder_weights": False,
    }
    if hasattr(model, "encoder") and hasattr(model.encoder, "policy_state"):
        payload["policy"] = model.encoder.policy_state()
    torch.save(payload, path)

    if "policy" in payload:
        policy_path = path.with_suffix(".policy.json")
        policy_path.write_text(json.dumps(payload["policy"], ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(model: nn.Module, path: str | Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    target_names = set(model.state_dict())
    dropped = [name for name in state if name not in target_names]
    bad_dropped = [name for name in dropped if not name.endswith("policy.gate_logits")]
    if bad_dropped:
        raise RuntimeError(f"Checkpoint tensors do not exist in target model: {bad_dropped[:20]}")
    filtered_state = {name: value for name, value in state.items() if name in target_names}
    incompatible = model.load_state_dict(filtered_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint tensors: {unexpected[:20]}")
    return payload
