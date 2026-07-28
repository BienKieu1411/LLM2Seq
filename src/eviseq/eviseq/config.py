"""Configuration and experiment-contract validation for EviSeq."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    if override.get("_replace_") is True:
        return {key: copy.deepcopy(value) for key, value in override.items() if key != "_replace_"}
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "_replace_":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parents(value: Any) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise TypeError("_base_ must be a path or a list of paths")


def _load(path: Path, stack: tuple[Path, ...]) -> Dict[str, Any]:
    path = path.resolve()
    if path in stack:
        raise ValueError(f"Cyclic config inheritance: {' -> '.join(map(str, (*stack, path)))}")
    with path.open("r", encoding="utf-8") as handle:
        own = yaml.safe_load(handle) or {}
    parent_value = own.pop("_base_", None)
    merged: Dict[str, Any] = {}
    for parent in _parents(parent_value):
        merged = _merge(merged, _load(path.parent / parent, (*stack, path)))
    return _merge(merged, own)


def architecture_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return fields that must stay fixed across datasets for a model variant."""

    return {
        "model": copy.deepcopy(config.get("model", {})),
        "native_attention": copy.deepcopy(config.get("native_attention", {})),
        "bridge": copy.deepcopy(config.get("bridge", {})),
        "decoder": copy.deepcopy(config.get("decoder", {})),
        "objectives": copy.deepcopy(config.get("objectives", {})),
    }


def inference_protocol_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fields that can change generated summaries without changing weights.

    Dataset file contents are bound separately by ``data_manifest.json``.  The
    fields below cover preprocessing, prompting, truncation and decoding so a
    paper-test cannot be rerun with a silently edited resolved config.
    """

    data = config.get("data", {})
    limits = config.get("limits", {})
    return {
        "data": {
            name: copy.deepcopy(data.get(name))
            for name in (
                "source_prefix",
                "sentence_separator",
                "decoder_instruction",
                "decoder_prefix",
                "use_decoder_chat_template",
                "enable_thinking",
                "clean_wikihow_metadata",
                "max_source_length",
                "max_target_length",
                "oracle_max_units",
            )
        },
        "generation": copy.deepcopy(config.get("generation", {})),
        "limits": {
            "max_validation_examples": int(limits.get("max_validation_examples", 0)),
            "max_test_examples": int(limits.get("max_test_examples", 0)),
        },
    }


def evaluation_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    """Complete frozen contract for generation from a trained checkpoint."""

    return {
        "architecture": architecture_contract(config),
        "inference_protocol": inference_protocol_contract(config),
    }


def _sha256(value: Dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def contract_sha256(config: Dict[str, Any]) -> str:
    return _sha256(architecture_contract(config))


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    config = _load(path, ())
    validate_config(config)
    meta = config.setdefault("_meta", {})
    meta["config_path"] = str(path)
    meta["architecture_sha256"] = contract_sha256(config)
    meta["inference_protocol_sha256"] = _sha256(inference_protocol_contract(config))
    meta["evaluation_contract_sha256"] = _sha256(evaluation_contract(config))
    return config


def validate_config(config: Dict[str, Any]) -> None:
    model = config.get("model", {})
    attention = config.get("native_attention", {})
    bridge = config.get("bridge", {})
    decoder = config.get("decoder", {})
    training = config.get("training", {})
    data = config.get("data", {})

    for field in ("encoder_name", "decoder_name"):
        if not str(model.get(field, "")).strip():
            raise ValueError(f"model.{field} is required")
    encoder_hidden = int(model.get("encoder_hidden_size", 0))
    decoder_hidden = int(model.get("decoder_hidden_size", 0))
    if encoder_hidden <= 0 or decoder_hidden <= 0:
        raise ValueError("model encoder/decoder hidden sizes must be positive")

    backend = str(attention.get("backend", "qwen_native"))
    if backend not in {"qwen_native", "pretrained_native"}:
        raise ValueError("native_attention.backend must be qwen_native or pretrained_native")
    variant = str(attention.get("variant", "evidence"))
    allowed = {"causal", "full", "dec2enc", "evidence", "pretrained"}
    if variant not in allowed:
        raise ValueError(f"native_attention.variant must be one of {sorted(allowed)}")
    if backend == "qwen_native" and variant == "pretrained":
        raise ValueError("qwen_native requires causal/full/dec2enc/evidence")
    if backend == "pretrained_native" and variant != "pretrained":
        raise ValueError("pretrained_native requires variant=pretrained")
    if backend == "qwen_native" and str(attention.get("implementation_revision", "")) != "evidence_key_v1":
        raise ValueError("qwen_native requires native_attention.implementation_revision=evidence_key_v1")
    implementation = str(model.get("encoder_attn_implementation", "sdpa"))
    if implementation not in {"sdpa", "flash_attention_2"}:
        raise ValueError("Only sdpa and flash_attention_2 are audited for the EviSeq encoder")
    if variant == "evidence" and implementation != "sdpa":
        raise ValueError("Evidence-key routing requires model.encoder_attn_implementation=sdpa")
    if backend == "qwen_native" and float(attention.get("evidence_key_bias_scale", 0.0)) <= 0.0:
        raise ValueError("native_attention.evidence_key_bias_scale must be positive")

    if int(bridge.get("salience_hidden_size", 0)) <= 0:
        raise ValueError("bridge.salience_hidden_size must be positive")
    if not 0.0 <= float(bridge.get("salience_gate_init", 0.0)) < 1.0:
        raise ValueError("bridge.salience_gate_init must be in [0, 1)")
    if int(decoder.get("cross_attention_every", 1)) != 1:
        raise ValueError("EviSeq requires copied cross-attention in every decoder layer")
    if int(decoder.get("memory_bank_count", 1)) != 1:
        raise ValueError("EviSeq uses one source memory, not HiRoute banks")

    objectives = config.get("objectives", {})
    if float(objectives.get("salience_weight", -1.0)) < 0.0:
        raise ValueError("objectives.salience_weight must be non-negative")
    use_contrastive = bool(objectives.get("use_contrastive", False))
    contrastive_weight = float(objectives.get("contrastive_weight", 0.0))
    if use_contrastive:
        if contrastive_weight <= 0.0:
            raise ValueError("Enabled prompt-source contrastive learning requires a positive weight")
        if float(objectives.get("contrastive_temperature", 0.0)) <= 0.0:
            raise ValueError("objectives.contrastive_temperature must be positive")
        if int(objectives.get("contrastive_projection_size", 0)) <= 0:
            raise ValueError("objectives.contrastive_projection_size must be positive")
        if str(objectives.get("contrastive_pooling", "")) not in {"mean", "mean_last"}:
            raise ValueError("objectives.contrastive_pooling must be mean or mean_last")
        if int(objectives.get("contrastive_warmup_epochs", -1)) < 0:
            raise ValueError("objectives.contrastive_warmup_epochs must be non-negative")
        if objectives.get("contrastive_across_accumulation") is not True:
            raise ValueError("Enabled contrastive learning requires exact virtual-batch GradCache")
    elif contrastive_weight != 0.0:
        raise ValueError("Disabled contrastive learning requires contrastive_weight=0")
    elif bool(objectives.get("contrastive_across_accumulation", False)):
        raise ValueError("Disabled contrastive learning cannot enable virtual-batch GradCache")
    forbidden = (
        "source_swap_weight",
        "response_alignment_weight",
        "phrase_mixture_weight",
        "label_smoothing",
    )
    if any(float(objectives.get(name, 0.0)) != 0.0 for name in forbidden):
        raise ValueError("EviSeq permits CE, salience, and target-free prompt-source InfoNCE only")

    if int(training.get("interface_warmup_epochs", -1)) < 0:
        raise ValueError("training.interface_warmup_epochs must be non-negative")
    if int(training.get("full_finetune_epochs", 0)) <= 0:
        raise ValueError("training.full_finetune_epochs must be positive")
    if int(training.get("batch_size", 0)) <= 0 or int(training.get("gradient_accumulation_steps", 0)) <= 0:
        raise ValueError("training batch and accumulation sizes must be positive")
    effective_contrastive_batch = int(training.get("batch_size", 0)) * int(
        training.get("gradient_accumulation_steps", 0)
    )
    if use_contrastive and effective_contrastive_batch < 2:
        raise ValueError("Prompt-source InfoNCE requires batch_size * gradient_accumulation_steps >= 2")
    if use_contrastive and int(objectives.get("contrastive_warmup_epochs", 0)) > int(
        training.get("interface_warmup_epochs", 0)
    ):
        raise ValueError("contrastive_warmup_epochs must finish within interface_warmup_epochs")

    for field in ("train_file", "validation_file", "test_file"):
        if not str(data.get(field, "")).strip():
            raise ValueError(f"data.{field} is required")
    if int(data.get("max_source_length", 0)) <= 0 or int(data.get("max_target_length", 0)) <= 1:
        raise ValueError("data lengths are invalid")

    checkpoint = config.get("checkpoint", {})
    if bool(checkpoint.get("save_best", False)) or bool(checkpoint.get("save_each_epoch", False)):
        raise ValueError("EviSeq saves last.pt only; best/epoch checkpoint selection is forbidden")
