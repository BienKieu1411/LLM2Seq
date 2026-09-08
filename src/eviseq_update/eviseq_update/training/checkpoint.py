"""Atomic AFMR checkpoints with structural, not path, compatibility."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ..distributed import rank, run_on_main, world_size


def architecture_spec(config: dict[str, Any]) -> dict[str, Any]:
    arch = config["architecture"]
    decoder = config["decoder"]
    spec = {
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
    copy_config = decoder.get("grounded_copy", {})
    if copy_config.get("enabled", False):
        spec["grounded_copy"] = {"alignment": "char_overlap_v1", "key_dim": int(copy_config.get("key_dim", 128))}
        semantic_config = copy_config.get("semantic_read", {})
        if semantic_config.get("enabled", False):
            spec["grounded_copy"]["semantic_read"] = {
                "graph": "shared_attention_residual_v1",
                "rank": int(semantic_config.get("rank", 128)),
            }
            attention = semantic_config.get("attention", "shared_copy")
            relative_rms = semantic_config.get("max_relative_rms")
            if attention != "shared_copy" or relative_rms is not None:
                spec["grounded_copy"]["semantic_read"].update(
                    graph="source_attention_residual_v2",
                    attention=attention,
                    max_relative_rms=None if relative_rms is None else float(relative_rms),
                )
    return spec


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    **metadata,
) -> None:
    """All training ranks participate; only rank zero writes the raw model."""
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() and world_size() == 1 else None,
        "cuda_current": torch.cuda.get_rng_state() if torch.cuda.is_available() and world_size() > 1 else None,
    }
    rng_states = [rng]
    if world_size() > 1:
        rng_states = [None] * world_size()
        dist.all_gather_object(rng_states, rng)
    run_on_main(lambda: _save_checkpoint(path, model, optimizer, config, rng_states=rng_states, **metadata))


def _save_checkpoint(
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
    elapsed_train_seconds: float | None = None,
    scheduler: Any = None,
    rng_states: list,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    training = config.get("training", {})
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "step": int(step),
        "best_metric": best_metric,
        "stage": stage,
        "stage_epoch": stage_epoch,
        "elapsed_train_seconds": None if elapsed_train_seconds is None else float(elapsed_train_seconds),
        "architecture_spec": architecture_spec(config),
        "training_spec": {
            **training,
            "decoder_attention_dropout": float(config["decoder"].get("attention_dropout", 0.0)),
            "world_size": len(rng_states),
            "effective_batch_size": (
                int(training["batch_size"]) * int(training["gradient_accumulation_steps"]) * len(rng_states)
                if "batch_size" in training and "gradient_accumulation_steps" in training
                else None
            ),
        },
        "rng_state": rng_states[0],
        "rng_states_by_rank": rng_states,
        "world_size": len(rng_states),
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
    rng_states = state.get("rng_states_by_rank")
    rng_state = rng_states[rank() % len(rng_states)] if rng_states else state.get("rng_state")
    if rng_state and restore_rng:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].to(device="cpu"))
        if rng_state.get("cuda_current") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["cuda_current"].to(device="cpu"))
        elif rng_state.get("cuda") is not None and torch.cuda.is_available():
            cuda_states = rng_state["cuda"]
            if world_size() > 1:
                torch.cuda.set_rng_state(cuda_states[torch.cuda.current_device() % len(cuda_states)].to("cpu"))
            else:
                torch.cuda.set_rng_state_all([value.to(device="cpu") for value in cuda_states])
    return {
        key: state.get(key)
        for key in (
            "epoch",
            "step",
            "best_metric",
            "stage",
            "stage_epoch",
            "elapsed_train_seconds",
            "architecture_spec",
            "training_spec",
            "world_size",
        )
    }
