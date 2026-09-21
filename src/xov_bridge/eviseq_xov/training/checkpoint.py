"""Atomic XOV checkpoints with structural, not path, compatibility."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module for DDP/DataParallel checkpoints."""

    while isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    return model


def architecture_spec(config: dict[str, Any]) -> dict[str, Any]:
    arch = config["architecture"]
    decoder = config["decoder"]
    spec = {
        "graph_version": "cross_tokenizer_ordered_value",
        "architecture": arch.get("name", "cross_tokenizer_ordered_value"),
        "bridge_mode": arch.get("bridge_mode", "cross_tokenizer_ordered_value"),
        "lexical_rank": int(arch.get("lexical_rank", 256)),
        "phrase_kernel": int(arch.get("phrase_kernel", 3)),
        "phrase_directional": bool(arch.get("phrase_directional", True)),
        "value_gate_max": float(arch.get("value_gate_max", 0.20)),
        "residual_reference_rms": float(arch.get("residual_reference_rms", 1.0)),
        "key_memory": "direct_projection",
        "copy_memory": "direct_projection",
        "cross_attention_every": int(decoder.get("cross_attention_every", 1)),
        "cross_gate_max": float(decoder.get("cross_gate_max", 1.0)),
    }
    copy_config = decoder.get("grounded_copy", {})
    if copy_config.get("enabled", False):
        spec["grounded_copy"] = {"alignment": "char_overlap_v1", "key_dim": int(copy_config.get("key_dim", 128))}
    return spec


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
) -> None:
    model = _unwrap_model(model)
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
        "elapsed_train_seconds": None if elapsed_train_seconds is None else float(elapsed_train_seconds),
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


def _embedding_shape_mismatches(
    checkpoint_model: dict[str, Any], model: torch.nn.Module
) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    """Return vocabulary-shape mismatches that usually indicate a wrong backbone.

    XOV checkpoints intentionally do not require model *paths* to stay the same:
    a local copy can move without invalidating its weights.  Embedding dimensions,
    however, are part of the trained graph.  Checking these tensors before the
    generic ``load_state_dict`` call turns a cryptic PyTorch error into an
    actionable encoder/decoder configuration error.
    """

    current_model = _unwrap_model(model).state_dict()
    mismatches = []
    for key in (
        "encoder.model.embed_tokens.weight",
        "decoder.backbone.embed_tokens.weight",
        "decoder.lm_head.weight",
    ):
        checkpoint_tensor = checkpoint_model.get(key)
        current_tensor = current_model.get(key)
        if checkpoint_tensor is None or current_tensor is None:
            continue
        checkpoint_shape = tuple(int(value) for value in checkpoint_tensor.shape)
        current_shape = tuple(int(value) for value in current_tensor.shape)
        if checkpoint_shape != current_shape:
            mismatches.append((key, checkpoint_shape, current_shape))
    return mismatches


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
    model = _unwrap_model(model)
    if config is not None and state.get("architecture_spec") != architecture_spec(config):
        raise ValueError("Checkpoint architecture_spec does not match the active XOV configuration")
    mismatches = _embedding_shape_mismatches(state["model"], model)
    if mismatches:
        details = "; ".join(
            f"{key}: checkpoint={checkpoint_shape}, active_model={current_shape}"
            for key, checkpoint_shape, current_shape in mismatches
        )
        raise ValueError(
            "Checkpoint/backbone vocabulary mismatch ("
            f"{details}). The checkpoint and evaluation config use different encoder or decoder tokenizers; "
            "for example, Qwen3-Embedding-0.6B has 151669 rows while Qwen3-0.6B has 151936. "
            "Evaluate with the resolved_config.yaml saved beside this checkpoint, and keep its "
            "model.encoder_name/model.decoder_name pair. Do not use strict=False or resize the embeddings."
        )
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
        key: state.get(key)
        for key in (
            "epoch",
            "step",
            "best_metric",
            "stage",
            "stage_epoch",
            "elapsed_train_seconds",
            "architecture_spec",
        )
    }
