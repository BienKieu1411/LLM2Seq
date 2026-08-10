"""Complete atomic EviSeq model checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

_RUNTIME_DATA_FIELDS = (
    "source_field",
    "target_field",
    "id_field",
    "source_template",
    "target_template",
    "list_separator",
    "source_prefix",
    "sentence_separator",
    "max_source_length",
    "decoder_instruction",
    "decoder_prefix",
    "use_decoder_chat_template",
    "enable_thinking",
    "clean_wikihow_metadata",
)
_RUNTIME_GENERATION_FIELDS = (
    "min_new_tokens",
    "max_new_tokens",
    "repetition_penalty",
    "no_repeat_ngram_size",
)


def evaluation_config_fingerprint(config: Dict[str, Any]) -> str:
    """Stable fingerprint for fields that change a checkpoint's decoded output.

    Batch size and dataset split deliberately do not participate.  In
    contrast, source serialization, model graph, and greedy constraints must
    agree with the checkpoint's resolved configuration.
    """

    data = config.get("data", {})
    generation = config.get("generation", {})
    material = {
        "model": config.get("model", {}),
        "native_attention": config.get("native_attention", {}),
        "bridge": config.get("bridge", {}),
        "decoder": config.get("decoder", {}),
        "data": {name: data.get(name) for name in _RUNTIME_DATA_FIELDS},
        "generation": {name: generation.get(name) for name in _RUNTIME_GENERATION_FIELDS},
    }
    return json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def assert_evaluation_config_matches_checkpoint(payload: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Fail closed when inference would use a different model/task protocol."""

    saved = payload.get("config")
    if not isinstance(saved, dict):
        raise RuntimeError("Checkpoint has no resolved configuration; refusing unverifiable evaluation")
    if evaluation_config_fingerprint(saved) != evaluation_config_fingerprint(config):
        raise RuntimeError(
            "Evaluation configuration differs from the checkpoint's resolved model, source serialization, "
            "or greedy-generation settings. Evaluate with that run's resolved_config.yaml instead."
        )


def _save_checkpoint(
    model: nn.Module,
    path: str | Path,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
    *,
    role: str,
) -> None:
    path = Path(path)
    if role not in {"last", "epoch", "best"}:
        raise ValueError(f"Unsupported checkpoint role: {role!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    payload = {
        "model_state_dict": state,
        "checkpoint_role": role,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_last_checkpoint(
    model: nn.Module,
    path: str | Path,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
) -> None:
    path = Path(path)
    if path.name != "last.pt":
        raise ValueError("The canonical final checkpoint must be named last.pt")
    _save_checkpoint(model, path, config, epoch, global_step, role="last")


def save_epoch_checkpoint(
    model: nn.Module,
    path: str | Path,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
) -> None:
    path = Path(path)
    expected_name = f"epoch_{int(epoch):03d}.pt"
    if path.name != expected_name:
        raise ValueError(f"Epoch {epoch} checkpoint must be named {expected_name}")
    _save_checkpoint(model, path, config, epoch, global_step, role="epoch")


def save_best_checkpoint(
    model: nn.Module,
    path: str | Path,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
) -> None:
    path = Path(path)
    if path.name != "best.pt":
        raise ValueError("The validation-selected checkpoint must be named best.pt")
    _save_checkpoint(model, path, config, epoch, global_step, role="best")


def save_configured_epoch_checkpoints(
    model: nn.Module,
    directory: str | Path,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
    validation_metrics: Dict[str, float] | None,
) -> Dict[str, Any]:
    """Save requested epoch artifacts and update validation-selected best.pt."""

    directory = Path(directory)
    checkpoint = config.get("checkpoint", {})
    result: Dict[str, Any] = {}
    if bool(checkpoint.get("save_each_epoch", False)):
        epoch_path = directory / f"epoch_{int(epoch):03d}.pt"
        save_epoch_checkpoint(model, epoch_path, config, epoch, global_step)
        result["epoch_path"] = str(epoch_path)

    if not bool(checkpoint.get("save_best", False)):
        return result
    if validation_metrics is None:
        raise RuntimeError("save_best=true requires validation metrics at every epoch")
    metric_name = str(checkpoint.get("best_metric", "eval_loss_ce"))
    if metric_name not in validation_metrics:
        raise KeyError(f"Best-checkpoint metric {metric_name!r} is absent from validation metrics")
    current = float(validation_metrics[metric_name])
    if not math.isfinite(current):
        raise RuntimeError(f"Best-checkpoint metric {metric_name!r} is non-finite: {current}")
    mode = str(checkpoint.get("best_mode", "min"))
    previous = getattr(model, "_best_checkpoint_value", None)
    improved = previous is None or (current < float(previous) if mode == "min" else current > float(previous))
    if improved:
        best_path = directory / "best.pt"
        save_best_checkpoint(model, best_path, config, epoch, global_step)
        model._best_checkpoint_value = current  # type: ignore[attr-defined]
        model._best_checkpoint_epoch = int(epoch)  # type: ignore[attr-defined]
        result.update(
            {
                "best_path": str(best_path),
                "best_metric": metric_name,
                "best_value": current,
            }
        )
    return result


def load_checkpoint(model: nn.Module, path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    role = payload.get("checkpoint_role")
    if role not in {"last", "epoch", "best"}:
        raise RuntimeError(f"Expected a complete EviSeq checkpoint, found role={role!r}")
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


def load_last_checkpoint(model: nn.Module, path: str | Path) -> Dict[str, Any]:
    payload = load_checkpoint(model, path)
    if payload.get("checkpoint_role") != "last":
        raise RuntimeError(f"Expected a last checkpoint, found {payload.get('checkpoint_role')!r}")
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
