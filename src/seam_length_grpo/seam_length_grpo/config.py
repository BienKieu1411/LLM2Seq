"""Pinned GRPO settings plus explicit word conditioning and reward settings."""

from __future__ import annotations

import math
from pathlib import Path

import yaml
from seam_grpo.config import DEFAULTS, cosine_multiplier

LENGTH_DEFAULTS = {
    "target_words_field": "target_words",
    "dataset_field": "dataset",
    "default_dataset": "default",
    "error_budget_field": "length_error_budget_words",
    "default_error_budget_words": 43.0,
    "dataset_error_budgets": {},
    "max_requested_words": 4096,
    "source_content_tokens": None,
    "encoder_prompt_reserve": 128,
    "decoder_prompt_reserve": 192,
    "condition_encoder": True,
    "condition_decoder": True,
    "encoder_instruction": "Đọc văn bản để chuẩn bị bản tóm tắt khoảng {target_words} từ. "
    "Ưu tiên các ý chính và chi tiết cần thiết để diễn đạt chính xác.\nVăn bản:\n",
    "decoder_instruction": "Viết bản tóm tắt khoảng {target_words} từ dựa trên văn bản đã cung cấp. "
    "Giữ các thông tin quan trọng, viết rõ ràng và chỉ trả về bản tóm tắt.",
}
SOURCE_DEFAULTS = {
    "weight": 0.05,
    "tau": 1.0,
    "reservoir_size": 256,
    "max_length_ratio": 2.0,
    "max_jaccard": 0.8,
    "teacher_checkpoint": None,
}
WARMUP_DEFAULTS = {"fraction": 0.05, "max_steps": 0}


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def extend(raw, path):
    """Validate new fields separately from the version-pinned training options."""
    for name, defaults in (
        ("length_control", LENGTH_DEFAULTS),
        ("source_reward", SOURCE_DEFAULTS),
        ("ce_warmup", WARMUP_DEFAULTS),
    ):
        supplied = raw.get(name, {})
        if not isinstance(supplied, dict) or set(supplied) - set(defaults):
            raise ValueError(f"Unknown or invalid {name} settings")
        raw[name] = {**defaults, **supplied}
    warmup = raw["ce_warmup"]
    fraction = warmup["fraction"]
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("ce_warmup.fraction must be a finite number in [0,1]")
    if isinstance(warmup["max_steps"], bool) or not isinstance(warmup["max_steps"], int) or warmup["max_steps"] < 0:
        raise ValueError("ce_warmup.max_steps must be a nonnegative integer")
    length, source = raw["length_control"], raw["source_reward"]
    for name in ("encoder_prompt_reserve", "decoder_prompt_reserve", "max_requested_words"):
        if isinstance(length[name], bool) or not isinstance(length[name], int) or length[name] < 1:
            raise ValueError(f"length_control.{name} must be a positive integer")
    if length["source_content_tokens"] is not None:
        positive(length["source_content_tokens"], "source_content_tokens")
        if not isinstance(length["source_content_tokens"], int):
            raise ValueError("source_content_tokens must be an integer")
    for name in ("condition_encoder", "condition_decoder"):
        if not isinstance(length[name], bool):
            raise ValueError(f"{name} must be boolean")
    for name in ("target_words_field", "dataset_field", "default_dataset", "error_budget_field"):
        if not isinstance(length[name], str) or not length[name].strip():
            raise ValueError(f"{name} must be nonempty text")
    from string import Formatter

    for name in ("encoder_instruction", "decoder_instruction"):
        template = length[name]
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"{name} must be nonempty text")
        fields = [field for _, field, spec, conversion in Formatter().parse(template) if field is not None]
        if not fields or any(field != "target_words" for field in fields):
            raise ValueError(f"{name} must use only the {{target_words}} field")
        template.format(target_words=100)
    positive(length["default_error_budget_words"], "default_error_budget_words")
    if not isinstance(length["dataset_error_budgets"], dict):
        raise ValueError("dataset_error_budgets must be a mapping")
    for domain, value in length["dataset_error_budgets"].items():
        if not isinstance(domain, str) or not domain:
            raise ValueError("dataset_error_budgets keys must be dataset names")
        positive(value, f"dataset_error_budgets.{domain}")
    positive(source["tau"], "source_reward.tau")
    positive(source["max_length_ratio"], "source_reward.max_length_ratio")
    if (
        isinstance(source["max_jaccard"], bool)
        or not isinstance(source["max_jaccard"], (int, float))
        or not math.isfinite(source["max_jaccard"])
    ):
        raise ValueError("source_reward.max_jaccard must be a finite number")
    if source["max_length_ratio"] < 1 or not 0 <= source["max_jaccard"] < 1:
        raise ValueError("Require max_length_ratio>=1 and max_jaccard in [0,1)")
    if (
        isinstance(source["reservoir_size"], bool)
        or not isinstance(source["reservoir_size"], int)
        or source["reservoir_size"] < 2
    ):
        raise ValueError("source_reward.reservoir_size must be an integer >=2")
    for name, value in (
        ("source_reward.weight", source["weight"]),
        ("reward.length_weight", raw["reward"].get("length_weight", 0.1)),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if source["teacher_checkpoint"]:
        p = Path(source["teacher_checkpoint"]).expanduser()
        source["teacher_checkpoint"] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    else:
        source["teacher_checkpoint"] = raw["sft_checkpoint"]
    return raw


def load(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("GRPO config must be a YAML mapping")
    allowed = {
        "base_config",
        "sft_checkpoint",
        "output_dir",
        "train_file",
        "validation_file",
        "training",
        "reward",
        "length_control",
        "source_reward",
        "ce_warmup",
    }
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
    raw["training"] = cfg
    raw.setdefault("reward", {})
    from .reward import LengthReward

    LengthReward(raw["reward"])
    return extend(raw, path)
