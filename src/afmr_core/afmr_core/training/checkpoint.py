"""Atomic checkpoints with strict architecture/config provenance."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ..config import config_fingerprint
from ..distributed import rank, run_on_main, world_size


def architecture_spec(config: dict[str, Any]) -> dict[str, Any]:
    """Return every graph choice that can change a tensor or probability path."""

    arch = config["architecture"]
    decoder = config["decoder"]
    copy_cfg = decoder["grounded_copy"]
    semantic = copy_cfg["semantic_read"]
    return {
        "graph_version": "afmr_core_value_anchor_dual_readout",
        "architecture": arch["name"],
        "architecture_config": {
            key: arch[key]
            for key in (
                "controller_dim",
                "depth_taps",
                "depth_rank",
                "depth_gate_init",
                "depth_gate_max",
                "feature_rank",
                "feature_gate_init",
                "feature_gate_max",
                "focus_hidden",
                "focus_windows",
                "focus_overlap",
                "focus_strength_init",
                "focus_strength_max",
                "temperature_init",
                "temperature_min",
                "temperature_max",
            )
        },
        "cross_attention": {
            "every": int(decoder["cross_attention_every"]),
            "initialize_cross_from_self": bool(decoder["initialize_cross_from_self"]),
            "gate_init": float(decoder["cross_gate_init"]),
            "gate_max": float(decoder["cross_gate_max"]),
            "query_cross_gate": bool(decoder.get("query_cross_gate", False)),
        },
        "grounded_copy": {
            "enabled": bool(copy_cfg["enabled"]),
            "alignment": "char_overlap",
            "key_dim": int(copy_cfg["key_dim"]),
            "gate_init": float(copy_cfg["gate_init"]),
            "readout_mode": copy_cfg["readout_mode"],
            "alpha_max": float(copy_cfg["alpha_max"]),
            "generate_reserve": float(copy_cfg["generate_reserve"]),
            "alpha_init": float(copy_cfg["alpha_init"]),
            "base_floor": float(copy_cfg["base_floor"]),
            "hidden_lambda": float(copy_cfg["hidden_lambda"]),
            "detach_copy_route_features": bool(copy_cfg.get("detach_copy_route_features", True)),
            "copy_entropy_feature": copy_cfg.get("copy_entropy_feature", "normalized_attention_entropy"),
            "hard_source_fallback": bool(copy_cfg.get("hard_source_fallback", True)),
            "null_slot": bool(copy_cfg.get("null_slot", False)),
            "logit_offset": copy_cfg["logit_offset"],
            "semantic": {
                "enabled": bool(semantic["enabled"]),
                "rank": int(semantic["rank"]),
                "num_heads": int(semantic["num_heads"]),
                "key_source": semantic["key_source"],
                "value_source": semantic["value_source"],
                "semantic_prior_scale": float(semantic["semantic_prior_scale"]),
                "max_relative_rms": float(semantic["max_relative_rms"]),
                "cap_mode": semantic.get("cap_mode", "smooth_relative_rms"),
                "inner_gate": bool(semantic["inner_gate"]),
                "output_init": semantic["output_init"],
            },
        },
    }


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
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
    elapsed_train_seconds: float | None = None,
    scheduler: Any = None,
    global_batch_manifest_hash: str | None = None,
    global_scheduler_step: int | None = None,
) -> None:
    """All ranks capture RNG; rank zero atomically writes one checkpoint."""

    local_rng = _capture_rng()
    rng_states = [local_rng]
    if world_size() > 1:
        rng_states = [None] * world_size()
        dist.all_gather_object(rng_states, local_rng)

    def write() -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        training = config.get("training", {})
        state = {
            "format_version": 2,
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
            "config_fingerprint": config_fingerprint(config),
            "training_spec": {
                **training,
                "world_size": len(rng_states),
                "effective_batch_size": int(training["batch_size"])
                * int(training["gradient_accumulation_steps"])
                * len(rng_states),
                "global_batch_manifest_hash": global_batch_manifest_hash,
                "global_scheduler_step": int(global_scheduler_step if global_scheduler_step is not None else step),
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

    run_on_main(write)


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _validate_training_protocol(state: dict[str, Any], config: dict[str, Any]) -> None:
    """Reject a resume whose distributed or optimizer protocol changed."""

    saved = state.get("training_spec")
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint has no training_spec for strict resume")
    saved_world = int(saved.get("world_size", state.get("world_size", 1)))
    if saved_world != world_size():
        raise ValueError(f"Checkpoint world_size={saved_world} does not match active world_size={world_size()}")
    training = config["training"]
    protocol_keys = (
        "interface_warmup_epochs",
        "full_finetune_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "length_bucketing",
        "length_bucket_multiplier",
        "weight_decay",
        "max_grad_norm",
        "seed",
        "scheduler",
        "warmup_steps",
    )
    for key in protocol_keys:
        if key in saved and saved[key] != training.get(key):
            raise ValueError(f"Checkpoint training protocol mismatch for {key}")
    expected_effective_batch = int(training["batch_size"]) * int(training["gradient_accumulation_steps"]) * world_size()
    if int(saved.get("effective_batch_size", expected_effective_batch)) != expected_effective_batch:
        raise ValueError("Checkpoint effective batch size does not match active training protocol")
    scheduler = state.get("scheduler")
    if scheduler is not None and int(scheduler.get("global_step", state.get("step", 0))) != int(state.get("step", 0)):
        raise ValueError("Checkpoint scheduler step does not match checkpoint optimizer step")


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
    validate_training_protocol: bool = False,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location=map_location, weights_only=False)
    if state.get("format_version", 1) < 2:
        raise ValueError("Checkpoint format is older than the AFMR provenance contract")
    if config is not None:
        expected_architecture = architecture_spec(config)
        if state.get("architecture_spec") != expected_architecture:
            raise ValueError("Checkpoint architecture_spec does not match the active AFMR configuration")
        expected_config = config_fingerprint(config)
        if state.get("config_fingerprint") != expected_config:
            raise ValueError("Checkpoint config_fingerprint does not match the active resolved configuration")
        if validate_training_protocol:
            _validate_training_protocol(state, config)
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    rng_states = state.get("rng_states_by_rank")
    if restore_rng:
        rng_state = rng_states[rank() % len(rng_states)] if rng_states else state.get("rng_state")
        if rng_state:
            _restore_rng(rng_state)
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
            "config_fingerprint",
            "training_spec",
            "world_size",
        )
    }


__all__ = ["architecture_spec", "load_checkpoint", "save_checkpoint"]
