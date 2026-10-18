"""Separate post-SFT settings; the saved SFT config supplies the architecture."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

DEFAULTS = {
    "epochs": 1,
    "prompt_batch_size": 1,
    "group_size": 4,
    "gradient_accumulation_steps": 4,
    "rollout_micro_batch_size": 4,
    "score_micro_batch_size": 1,
    "logprob_chunk_size": 32,
    "max_new_tokens": None,
    "temperature": 1.0,
    "epsilon": 0.2,
    "scale_rewards": "none",
    "ce_weight": 0.05,
    "beta": 0.0,
    "encoder_lr": 1e-6,
    "decoder_lr": 1e-6,
    "bridge_lr": 3e-6,
    "cross_attention_lr": 3e-6,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "min_lr_ratio": 0.1,
    "max_grad_norm": 1.0,
    "seed": 42,
    "num_workers": 0,
    "max_steps": 0,
    "validate_every_steps": 0,
    "max_train_examples": 0,
    "validation_max_examples": 0,
    "validation_batch_size": 4,
    "log_every_steps": 1,
    "save_each_epoch": True,
    "save_best": True,
    "evaluate_before_training": True,
    "rollout_backend": "native",
    "vllm_gpu_memory_utilization": 0.20,
    "vllm_max_num_seqs": 16,
    "vllm_max_num_batched_tokens": 2048,
    "vllm_encoder_max_batch_tokens": 8192,
    "vllm_logp_mean_tolerance": 0.05,
    "vllm_logp_max_tolerance": 0.50,
    "vllm_group_sampling": False,
    "epsilon_high": None,
    "delta": None,
    "importance_sampling_level": "token",
    "loss_type": "dr_grpo",
    "top_entropy_quantile": 1.0,
    "entropy_coef": 0.0,
    "off_policy_mask_threshold": None,
}


def load(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("GRPO config must be a YAML mapping")
    allowed = {"base_config", "sft_checkpoint", "output_dir", "train_file", "validation_file", "training", "reward"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown GRPO keys: {sorted(set(raw) - allowed)}")
    for key in ("base_config", "sft_checkpoint", "output_dir"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"{key} is required")
    for key in ("base_config", "sft_checkpoint", "output_dir", "train_file", "validation_file"):
        if raw.get(key):
            p = Path(raw[key]).expanduser()
            raw[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    supplied = raw.get("training", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("Invalid / unknown GRPO training settings")
    cfg = {**DEFAULTS, **supplied}
    integers = (
        "epochs",
        "prompt_batch_size",
        "group_size",
        "gradient_accumulation_steps",
        "rollout_micro_batch_size",
        "score_micro_batch_size",
        "logprob_chunk_size",
        "validation_batch_size",
        "log_every_steps",
        "vllm_max_num_seqs",
        "vllm_max_num_batched_tokens",
        "vllm_encoder_max_batch_tokens",
    )
    for key in integers:
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in (
        "seed",
        "num_workers",
        "max_steps",
        "max_train_examples",
        "validation_max_examples",
        "validate_every_steps",
    ):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError(f"training.{key} must be a nonnegative integer")
    if cfg["group_size"] < 2:
        raise ValueError("GRPO needs group_size >= 2")
    if cfg["max_new_tokens"] is not None and (
        isinstance(cfg["max_new_tokens"], bool)
        or not isinstance(cfg["max_new_tokens"], int)
        or cfg["max_new_tokens"] < 1
    ):
        raise ValueError("max_new_tokens must be null or a positive integer")
    for key in ("temperature", "encoder_lr", "decoder_lr", "bridge_lr", "cross_attention_lr"):
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"training.{key} must be positive and finite")
    for key in ("ce_weight", "beta", "weight_decay", "max_grad_norm"):
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"training.{key} must be nonnegative and finite")
    for key in ("warmup_ratio", "min_lr_ratio"):
        cfg[key] = float(cfg[key])
        if not 0 <= cfg[key] < 1:
            raise ValueError(f"training.{key} must be in [0, 1)")
    cfg["epsilon"] = float(cfg["epsilon"])
    if not 0 < cfg["epsilon"] < 1:
        raise ValueError("epsilon must be in (0, 1)")
    if cfg["scale_rewards"] not in {"none", "group"}:
        raise ValueError("scale_rewards must be none or group")
    for key in ("save_each_epoch", "save_best", "evaluate_before_training", "vllm_group_sampling"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"training.{key} must be boolean")
    if cfg["rollout_backend"] not in {"native", "seam_vllm"}:
        raise ValueError("rollout_backend must be native or seam_vllm")
    cfg["vllm_gpu_memory_utilization"] = float(cfg["vllm_gpu_memory_utilization"])
    if not 0 < cfg["vllm_gpu_memory_utilization"] < 1:
        raise ValueError("vllm_gpu_memory_utilization must be in (0, 1)")
    for key in ("vllm_logp_mean_tolerance", "vllm_logp_max_tolerance"):
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if cfg["epsilon_high"] is not None:
        cfg["epsilon_high"] = float(cfg["epsilon_high"])
        if not 0 < cfg["epsilon_high"] < 1:
            raise ValueError("epsilon_high must be in (0, 1)")
    if cfg["delta"] is not None:
        cfg["delta"] = float(cfg["delta"])
        if cfg["delta"] <= 0 or not math.isfinite(cfg["delta"]):
            raise ValueError("delta must be positive and finite")
    if cfg["importance_sampling_level"] not in {"token", "sequence"}:
        raise ValueError("importance_sampling_level must be 'token' or 'sequence'")
    allowed_loss_types = {"grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo", "vespo"}
    if cfg["loss_type"] not in allowed_loss_types:
        raise ValueError(f"loss_type must be one of {sorted(allowed_loss_types)}")
    cfg["top_entropy_quantile"] = float(cfg["top_entropy_quantile"])
    if not 0 < cfg["top_entropy_quantile"] <= 1.0:
        raise ValueError("top_entropy_quantile must be in (0, 1]")
    cfg["entropy_coef"] = float(cfg["entropy_coef"])
    if cfg["entropy_coef"] < 0 or not math.isfinite(cfg["entropy_coef"]):
        raise ValueError("entropy_coef must be nonnegative and finite")
    if cfg["off_policy_mask_threshold"] is not None:
        cfg["off_policy_mask_threshold"] = float(cfg["off_policy_mask_threshold"])
        if cfg["off_policy_mask_threshold"] <= 0 or not math.isfinite(cfg["off_policy_mask_threshold"]):
            raise ValueError("off_policy_mask_threshold must be positive and finite")
    raw["training"] = cfg
    raw.setdefault("reward", {})
    return raw


def cosine_multiplier(step: int, total: int, warmup: int, floor: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
