"""Atomic AFMR checkpoints with structural, not path, compatibility."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def architecture_spec(config: dict[str, Any]) -> dict[str, Any]:
    arch = config["architecture"]
    decoder = config["decoder"]
    return {
        "graph_version": "afmr_token_depth_lowrank_v3",
        "architecture": arch.get("name"),
        "controller_dim": int(arch.get("controller_dim", 0)),
        "depth_taps": int(arch.get("depth_taps", 0)),
        "depth_rank": int(arch.get("depth_rank", 0)),
        "depth_gate_max": float(arch.get("depth_gate_max", 0.0)),
        "feature_rank": int(arch.get("feature_rank", 0)),
        "feature_gate_max": float(arch.get("feature_gate_max", 0.0)),
        "focus_hidden": int(arch.get("focus_hidden", 0)),
        "focus_windows": tuple(int(value) for value in arch.get("focus_windows", ())),
        "focus_overlap": float(arch.get("focus_overlap", 0.0)),
        "focus_strength_max": float(arch.get("focus_strength_max", 0.0)),
        "temperature_min": float(arch.get("temperature_min", 0.0)),
        "temperature_max": float(arch.get("temperature_max", 0.0)),
        "cross_attention_every": int(decoder.get("cross_attention_every", 1)),
        "cross_gate_max": float(decoder.get("cross_gate_max", 0.0)),
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    *,
    epoch: int,
    step: int,
    best_metric: float | None = None,
    stage: str | None = None,
    stage_epoch: int | None = None,
    scheduler: Any = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "step": int(step),
        "best_metric": best_metric,
        "stage": stage,
        "stage_epoch": stage_epoch,
        "architecture_spec": architecture_spec(config),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    config: dict[str, Any] | None = None,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    scheduler: Any = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if config is not None and state.get("architecture_spec") != architecture_spec(config):
        raise ValueError("Checkpoint architecture_spec does not match the active AFMR configuration")
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    rng_state = state.get("rng_state")
    if rng_state and restore_rng:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].to(device="cpu"))
        if rng_state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.to(device="cpu") for value in rng_state["cuda"]])
    return {
        key: state.get(key) for key in ("epoch", "step", "best_metric", "stage", "stage_epoch", "architecture_spec")
    }
