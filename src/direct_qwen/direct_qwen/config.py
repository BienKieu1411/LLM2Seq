"""Fail-closed configuration for the direct Qwen3-0.6B control."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

SRC_ROOT = Path(__file__).resolve().parents[2]

_DATA_FIELDS = (
    "train_file",
    "validation_file",
    "test_file",
    "source_prefix",
    "sentence_separator",
    "decoder_instruction",
    "decoder_prefix",
    "use_decoder_chat_template",
    "enable_thinking",
    "clean_wikihow_metadata",
    "max_source_length",
    "max_target_length",
)
_GENERATION_FIELDS = (
    "batch_size",
    "min_new_tokens",
    "max_new_tokens",
    "repetition_penalty",
    "no_repeat_ngram_size",
)


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_reference_config(raw: Dict[str, Any]) -> Path:
    value = str(raw.get("contract", {}).get("reference_config", "")).strip()
    if not value:
        raise ValueError("contract.reference_config is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SRC_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen EviSeq config not found: {path}")
    return path


def shared_reference_contract(reference: Dict[str, Any]) -> Dict[str, Any]:
    """Fields that must be identical for a controlled external baseline."""

    return {
        "data": {name: copy.deepcopy(reference["data"][name]) for name in _DATA_FIELDS},
        "generation": {name: copy.deepcopy(reference["generation"][name]) for name in _GENERATION_FIELDS},
        "limits": copy.deepcopy(reference.get("limits", {})),
    }


def direct_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    """Serializable training and inference contract stored with the checkpoint."""

    return {
        "base_model_id": config["model"]["base_model_id"],
        "prompt_layout": config["contract"]["prompt_layout"],
        "eviseq_shared_contract_sha256": config["contract"]["eviseq_shared_contract_sha256"],
        "model": copy.deepcopy(config["model"]),
        "training": copy.deepcopy(config["training"]),
        "data": {name: copy.deepcopy(config["data"][name]) for name in _DATA_FIELDS},
        "generation": {name: copy.deepcopy(config["generation"][name]) for name in _GENERATION_FIELDS},
        "checkpoint": copy.deepcopy(config["checkpoint"]),
    }


def validate_config(config: Dict[str, Any], reference: Dict[str, Any]) -> None:
    contract = config.get("contract", {})
    model = config.get("model", {})
    training = config.get("training", {})
    checkpoint = config.get("checkpoint", {})

    if contract.get("prompt_layout") != "eviseq_source_ids_then_eviseq_decoder_seed":
        raise ValueError("The direct baseline must preserve both frozen EviSeq prompt segments")
    if contract.get("preserve_source_terminal_eos") is not True:
        raise ValueError("The EviSeq source token segment, including terminal EOS, must be preserved")
    expected_architecture = str(contract.get("eviseq_architecture_sha256", ""))
    if expected_architecture != str(reference.get("_meta", {}).get("architecture_sha256", "")):
        raise ValueError("The referenced EviSeq architecture is not the frozen paper configuration")
    shared = shared_reference_contract(reference)
    shared_hash = _sha256(shared)
    if str(contract.get("eviseq_shared_contract_sha256", "")) != shared_hash:
        raise ValueError("EviSeq data/prompt/generation contract drifted; freeze a new controlled experiment")

    if str(model.get("base_model_id", "")) != "Qwen/Qwen3-0.6B":
        raise ValueError("This control is specifically Qwen/Qwen3-0.6B")
    if model.get("local_files_only") is not True:
        raise ValueError("model.local_files_only must be true; downloads are forbidden")
    if str(model.get("torch_dtype", "")).lower() not in {"float32", "fp32"}:
        raise ValueError("Full fine-tuning requires FP32 master parameters")
    if int(model.get("minimum_context_length", 0)) < 4096:
        raise ValueError("The direct model must fit the independent 3072/384 EviSeq token budgets")

    expected_epochs = int(reference["training"]["interface_warmup_epochs"]) + int(
        reference["training"]["full_finetune_epochs"]
    )
    if str(training.get("mode", "")) != "full_finetune":
        raise ValueError("training.mode must be full_finetune")
    if int(training.get("num_train_epochs", 0)) != expected_epochs:
        raise ValueError(f"Direct Qwen must see the same {expected_epochs} total epochs as EviSeq")
    if int(training.get("batch_size", 0)) != int(reference["training"]["batch_size"]):
        raise ValueError("Physical batch size must match EviSeq")
    if int(training.get("gradient_accumulation_steps", 0)) != int(reference["training"]["gradient_accumulation_steps"]):
        raise ValueError("Gradient accumulation must match EviSeq")
    if float(training.get("learning_rate", 0.0)) != float(reference["training"]["full_decoder_lr"]):
        raise ValueError("Direct Qwen LR must match the EviSeq pretrained decoder LR")
    if not bool(training.get("bf16", False)) or bool(training.get("fp16", False)):
        raise ValueError("The frozen precision contract is FP32 master weights with BF16 autocast")

    if checkpoint.get("save_last") is not True:
        raise ValueError("The direct baseline must save last.pt")
    if bool(checkpoint.get("save_best", False)) or bool(checkpoint.get("save_each_epoch", False)):
        raise ValueError("Best and per-epoch checkpoints are forbidden")
    if "huggingface" in config:
        raise ValueError("Hub configuration is forbidden in this local-only baseline")


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load direct settings and inject the authoritative EviSeq shared contract."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("Direct Qwen config must contain a YAML mapping")

    reference_path = _resolve_reference_config(raw)
    loader_value = str(raw.get("contract", {}).get("reference_loader", "")).strip()
    if ":" not in loader_value:
        raise ValueError("contract.reference_loader must be 'module:function'")
    module_name, function_name = loader_value.split(":", maxsplit=1)
    reference_loader = getattr(importlib.import_module(module_name), function_name)
    reference = reference_loader(reference_path)
    config = copy.deepcopy(raw)
    shared = shared_reference_contract(reference)
    config["data"] = shared["data"]
    config["generation"] = shared["generation"]
    config["limits"] = shared["limits"]
    config["benchmark"] = copy.deepcopy(reference.get("benchmark", {}))
    override = os.environ.get("DIRECT_QWEN_MODEL_PATH", "").strip()
    if override:
        config.setdefault("model", {})["name_or_path"] = override
    validate_config(config, reference)
    config["_meta"] = {
        "config_path": str(config_path),
        "reference_config_path": str(reference_path),
        "eviseq_architecture_sha256": reference["_meta"]["architecture_sha256"],
        "eviseq_shared_contract_sha256": _sha256(shared),
    }
    config["_meta"]["direct_contract_sha256"] = _sha256(direct_contract(config))
    return config
