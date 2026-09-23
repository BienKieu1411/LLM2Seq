"""Atomic XOV checkpoints with structural, not path, compatibility."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module for DDP/DataParallel checkpoints."""

    while isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    return model


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def collect_rank_rng_states() -> list[dict[str, Any]]:
    local = capture_rng_state()
    if not (dist.is_available() and dist.is_initialized()):
        return [local]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, local)
    return states


def _asset_fingerprint(name: str) -> str:
    if name == "__tiny__":
        return "tiny_qwen_fixture"
    root = Path(name)
    if not root.is_dir():
        raise ValueError(f"XOV requires a local model directory: {name}")
    digest = hashlib.sha256()
    # No model weights are read. Moving an unchanged local model remains valid.
    names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
    )
    for filename in names:
        path = root / filename
        if path.is_file():
            digest.update(filename.encode())
            if filename == "config.json":
                import json

                config = json.loads(path.read_text())
                for key in ("_name_or_path", "transformers_version"):
                    config.pop(key, None)
                digest.update(json.dumps(config, sort_keys=True).encode())
            else:
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
    return digest.hexdigest()


def architecture_spec(config: dict[str, Any]) -> dict[str, Any]:
    arch = config["architecture"]
    decoder = config["decoder"]
    spec = {
        "graph_version": "ordered_lexical_values",
        "operator_contract": "silu_before_pool_original_adjacency_clipped_source_copy_anchor",
        "input_policy": {
            key: config["data"].get(key, default)
            for key, default in {
                "encoder_prefix": "",
                "max_source_length": 4096,
                "max_target_length": 512,
                "decoder_prompt": "",
                "decoder_prefix": "",
                "decoder_chat_template": False,
                "source_field": "text",
                "target_field": "summary",
                "list_separator": "\n",
                "detokenize": False,
            }.items()
        },
        "execution_policy": {
            "tokenizer_use_fast": bool(config["model"].get("tokenizer_use_fast", True)),
            "trust_remote_code": bool(config["model"].get("trust_remote_code", True)),
            "attention_implementation": config["model"].get("attention_implementation", "sdpa"),
            "attention_dropout": float(decoder.get("attention_dropout", 0.0)),
        },
        "encoder_assets": _asset_fingerprint(str(config["model"]["encoder_name"])),
        "decoder_assets": _asset_fingerprint(str(config["model"]["decoder_name"])),
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
        spec["grounded_copy"] = {
            "alignment": "clipped_visible_overlap",
            "key_dim": int(copy_config.get("key_dim", 128)),
        }
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
    rank_rng_states: list[dict[str, Any]] | None = None,
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
        "rank_rng_states": rank_rng_states if rank_rng_states is not None else [capture_rng_state()],
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
    state = torch.load(Path(path), map_location=map_location, weights_only=False)
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
    rng_states = state.get("rank_rng_states")
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    if restore_rng and (not rng_states or len(rng_states) != world_size):
        raise ValueError("Exact RNG resume requires the original world size and per-rank RNG states")
    rng_state = rng_states[dist.get_rank() if distributed else 0] if restore_rng else None
    if rng_state is not None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].to(device="cpu"))
        if rng_state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["cuda"].to(device="cpu"))
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
