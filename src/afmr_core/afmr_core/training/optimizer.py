"""Explicit parameter groups for the AFMR training protocol."""

from __future__ import annotations

from typing import Any

import torch


def _component(name: str) -> str:
    """Classify a parameter by graph ownership, never by ``requires_grad``."""

    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("bridge."):
        return "bridge"
    if name.startswith("decoder.grounded_copy."):
        return "grounded_copy"
    if name.startswith("decoder.semantic_reader."):
        return "semantic_read"
    if name.startswith("decoder.dual_readout."):
        return "dual_readout"
    if name.startswith("decoder.backbone.layers.") and any(
        marker in name for marker in (".cross.", ".cross_norm.", ".cross_gate")
    ):
        return "cross_attention"
    if name.startswith("decoder."):
        return "decoder"
    raise ValueError(f"Unclassified AFMR parameter: {name}")


def set_stage_trainability(model: torch.nn.Module, stage: str) -> None:
    """Set warm-up/full trainability using the named graph components."""

    if stage not in {"interface_warmup", "full_finetune"}:
        raise ValueError(f"Unknown AFMR training stage: {stage}")
    warmup = {"bridge", "cross_attention", "grounded_copy", "semantic_read", "dual_readout"}
    for name, parameter in model.named_parameters():
        parameter.requires_grad = stage == "full_finetune" or _component(name) in warmup


def build_optimizer(model: torch.nn.Module, config: dict[str, Any], stage: str) -> torch.optim.Optimizer:
    if stage not in {"interface_warmup", "full_finetune"}:
        raise ValueError(f"Unknown AFMR training stage: {stage}")
    training = config["training"]
    prefix = "warmup" if stage == "interface_warmup" else "full"
    groups: dict[tuple[str, bool], dict[str, Any]] = {}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"Duplicate optimizer parameter: {name}")
        seen.add(id(parameter))
        component = _component(name)
        lr_key = f"{prefix}_{component}_lr"
        if lr_key not in training:
            raise ValueError(f"Missing optimizer learning rate: training.{lr_key}")
        decay = parameter.ndim > 1
        key = (component, decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "name": component + ("_decay" if decay else "_no_decay"),
                "component": component,
                "lr": float(training[lr_key]),
                "initial_lr": float(training[lr_key]),
                "weight_decay": float(training.get("weight_decay", 0.01)) if decay else 0.0,
            }
        groups[key]["params"].append(parameter)
    if not groups:
        raise ValueError(f"No trainable parameters for stage {stage}")
    fused = bool(training.get("fused_optimizer", False)) and all(
        parameter.device.type == "cuda" for group in groups.values() for parameter in group["params"]
    )
    kwargs = {"betas": (0.9, 0.95), "eps": 1e-8}
    if fused:
        try:
            return torch.optim.AdamW(list(groups.values()), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            # Some torch/CUDA combinations expose AdamW but not fused kernels.
            pass
    return torch.optim.AdamW(list(groups.values()), **kwargs)


__all__ = ["_component", "build_optimizer", "set_stage_trainability"]
