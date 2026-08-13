"""Configuration loading and validation for EviSeq."""

from __future__ import annotations

import copy
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


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    config = _load(path, ())
    validate_config(config)
    meta = config.setdefault("_meta", {})
    meta["config_path"] = str(path)
    return config


def resolve_data_path(value: str | Path, config: Dict[str, Any]) -> Path:
    """Resolve configured JSONL paths across repo and flattened Colab layouts."""

    path = Path(value)
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    config_path = config.get("_meta", {}).get("config_path")
    if config_path:
        candidates.append(Path(str(config_path)).resolve().parent / path)

    package_root = Path(__file__).resolve().parents[1]
    if path.parts[:2] == ("src", "eviseq"):
        candidates.append(package_root.joinpath(*path.parts[2:]))
    elif path.parts[:1] == ("eviseq",):
        candidates.append(package_root.joinpath(*path.parts[1:]))

    candidates.extend([Path.cwd() / path, package_root / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def validate_config(config: Dict[str, Any]) -> None:
    model = config.get("model", {})
    attention = config.get("native_attention", {})
    bridge = config.get("bridge", {})
    decoder = config.get("decoder", {})
    training = config.get("training", {})
    data = config.get("data", {})
    task = config.get("task", {})

    if str(task.get("format", "text_to_text")) != "text_to_text":
        raise ValueError("task.format must be text_to_text")
    metrics = task.get("metrics", ["rouge"])
    if not isinstance(metrics, list):
        raise ValueError("task.metrics must be a list")
    supported_metrics = {"rouge", "exact_match", "token_f1"}
    unknown_metrics = set(map(str, metrics)) - supported_metrics
    if unknown_metrics:
        raise ValueError(f"Unsupported task metrics: {sorted(unknown_metrics)}")
    metric_callable = str(task.get("metric_callable", "")).strip()
    if metric_callable and ":" not in metric_callable:
        raise ValueError("task.metric_callable must use module:function syntax")
    if not metrics and not metric_callable:
        raise ValueError("Configure at least one built-in metric or task.metric_callable")

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
    if backend == "qwen_native" and str(attention.get("implementation_revision", "")) != "selective_evidence_v2":
        raise ValueError("qwen_native requires native_attention.implementation_revision=selective_evidence_v2")
    bidirectional_layers = int(attention.get("bidirectional_layer_count", 0))
    if backend == "qwen_native" and variant != "causal" and bidirectional_layers <= 0:
        raise ValueError("Non-causal qwen_native variants require native_attention.bidirectional_layer_count > 0")
    if backend == "qwen_native" and bidirectional_layers < 0:
        raise ValueError("native_attention.bidirectional_layer_count must be non-negative")
    implementation = str(model.get("encoder_attn_implementation", "sdpa"))
    if implementation not in {"sdpa", "flash_attention_2"}:
        raise ValueError("Only sdpa and flash_attention_2 are supported for the EviSeq encoder")
    if variant == "evidence" and implementation != "sdpa":
        raise ValueError("Evidence-key routing requires model.encoder_attn_implementation=sdpa")
    if backend == "qwen_native" and float(attention.get("evidence_key_bias_scale", 0.0)) <= 0.0:
        raise ValueError("native_attention.evidence_key_bias_scale must be positive")
    gate_init = float(attention.get("evidence_view_gate_init", 0.0))
    if not 0.0 <= gate_init < 1.0:
        raise ValueError("native_attention.evidence_view_gate_init must be in [0, 1)")

    if int(bridge.get("salience_hidden_size", 0)) <= 0:
        raise ValueError("bridge.salience_hidden_size must be positive")
    if not isinstance(bridge.get("trainable_identity_projection", False), bool):
        raise ValueError("bridge.trainable_identity_projection must be boolean")
    if bool(bridge.get("trainable_identity_projection", False)) and encoder_hidden != decoder_hidden:
        raise ValueError("bridge.trainable_identity_projection requires equal encoder and decoder hidden sizes")
    gate_parameterization = str(bridge.get("salience_gate_parameterization", "signed_tanh"))
    if gate_parameterization not in {"signed_tanh", "sigmoid"}:
        raise ValueError("bridge.salience_gate_parameterization must be signed_tanh or sigmoid")
    if not 0.0 <= float(bridge.get("salience_gate_init", 0.0)) < 1.0:
        raise ValueError("bridge.salience_gate_init must be in [0, 1)")
    if gate_parameterization == "sigmoid" and float(bridge.get("salience_gate_init", 0.0)) <= 0.0:
        raise ValueError("sigmoid bridge.salience_gate_init must be in (0, 1)")
    if float(bridge.get("salience_bias_scale", 0.0)) <= 0.0:
        raise ValueError("bridge.salience_bias_scale must be positive")
    length_normalization = str(bridge.get("salience_length_normalization", "legacy_gated"))
    if length_normalization not in {"legacy_gated", "unit_invariant"}:
        raise ValueError("bridge.salience_length_normalization must be legacy_gated or unit_invariant")
    if float(bridge.get("salience_ranking_weight", 0.0)) < 0.0:
        raise ValueError("bridge.salience_ranking_weight must be non-negative")
    if int(decoder.get("cross_attention_every", 1)) != 1:
        raise ValueError("EviSeq requires copied cross-attention in every decoder layer")
    if int(decoder.get("memory_bank_count", 1)) != 1:
        raise ValueError("EviSeq uses one source memory, not HiRoute banks")

    objectives = config.get("objectives", {})
    if float(objectives.get("salience_weight", -1.0)) < 0.0:
        raise ValueError("objectives.salience_weight must be non-negative")

    # Optional document-level contrastive objective.
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

    # Evidence-focused contrastive objective.
    use_evidence_contrastive = bool(objectives.get("use_evidence_contrastive", True))
    evi_weight = float(objectives.get("evidence_contrastive_weight", 0.05))
    if use_evidence_contrastive:
        if data.get("supervise_evidence", True) is not True:
            raise ValueError("Evidence contrastive requires data.supervise_evidence=true")
        if evi_weight <= 0.0:
            raise ValueError("Enabled evidence contrastive requires a positive weight")
        if float(objectives.get("evidence_contrastive_temperature", 0.07)) <= 0.0:
            raise ValueError("objectives.evidence_contrastive_temperature must be positive")
        if int(objectives.get("evidence_hard_negatives", 2)) <= 0:
            raise ValueError("objectives.evidence_hard_negatives must be positive")
        if int(objectives.get("evidence_hard_negatives_full", objectives.get("evidence_hard_negatives", 2))) <= 0:
            raise ValueError("objectives.evidence_hard_negatives_full must be positive")
        if float(objectives.get("evidence_hard_negative_salience_boost", 0.1)) < 0.0:
            raise ValueError("objectives.evidence_hard_negative_salience_boost must be non-negative")
        if int(objectives.get("evidence_contrastive_projection_size", 128)) <= 0:
            raise ValueError("objectives.evidence_contrastive_projection_size must be positive")
        evidence_mode = str(objectives.get("evidence_contrastive_mode", "document"))
        if evidence_mode not in {"document", "sentence_aligned"}:
            raise ValueError("objectives.evidence_contrastive_mode must be document or sentence_aligned")
        if float(objectives.get("evidence_contrastive_salience_bias", 0.0)) < 0.0:
            raise ValueError("objectives.evidence_contrastive_salience_bias must be non-negative")
        if evidence_mode == "sentence_aligned" and data.get("sentence_evidence_supervision", False) is not True:
            raise ValueError("sentence_aligned evidence contrastive requires data.sentence_evidence_supervision=true")
        if (
            evidence_mode == "sentence_aligned"
            and float(objectives.get("evidence_contrastive_salience_bias", 0.0)) > 0.0
            and data.get("sentence_evidence_use_union_as_salience", False) is not True
        ):
            raise ValueError(
                "Sentence-aligned salience coupling requires data.sentence_evidence_use_union_as_salience=true"
            )
        if (
            data.get("sentence_evidence_use_union_as_salience", False)
            and data.get("sentence_evidence_supervision", False) is not True
        ):
            raise ValueError("sentence_evidence_use_union_as_salience requires sentence_evidence_supervision=true")
        evi_warmup = int(objectives.get("evidence_contrastive_warmup_epochs", 0))
        if evi_warmup < 0:
            raise ValueError("objectives.evidence_contrastive_warmup_epochs must be non-negative")
    elif evi_weight != 0.0:
        raise ValueError("Disabled evidence contrastive requires evidence_contrastive_weight=0")
    if data.get("supervise_evidence", True) is not True and float(objectives.get("salience_weight", 0.0)) != 0.0:
        raise ValueError("Tasks without evidence supervision require objectives.salience_weight=0")

    forbidden = (
        "source_swap_weight",
        "response_alignment_weight",
        "phrase_mixture_weight",
        "label_smoothing",
    )
    if any(float(objectives.get(name, 0.0)) != 0.0 for name in forbidden):
        raise ValueError("EviSeq permits CE, salience, evidence contrastive, and optional document InfoNCE only")

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
    if use_evidence_contrastive and int(objectives.get("evidence_contrastive_warmup_epochs", 0)) > int(
        training.get("interface_warmup_epochs", 0)
    ):
        raise ValueError("evidence_contrastive_warmup_epochs must finish within interface_warmup_epochs")

    for field in ("train_file", "validation_file"):
        if not str(data.get(field, "")).strip():
            raise ValueError(f"data.{field} is required")
    if int(data.get("max_source_length", 0)) <= 0 or int(data.get("max_target_length", 0)) <= 1:
        raise ValueError("data lengths are invalid")
    if int(data.get("sentence_evidence_max_units", 1)) <= 0:
        raise ValueError("data.sentence_evidence_max_units must be positive")

    online_kd = config.get("online_kd", {})
    if online_kd and bool(online_kd.get("enabled", False)):
        if int(online_kd.get("epochs", 1)) <= 0:
            raise ValueError("online_kd.epochs must be positive")
        if float(online_kd.get("weight", 0.1)) <= 0.0:
            raise ValueError("online_kd.weight must be positive")
        if float(online_kd.get("temperature", 2.0)) <= 0.0:
            raise ValueError("online_kd.temperature must be positive")
        if int(online_kd.get("topk", 32)) <= 0:
            raise ValueError("online_kd.topk must be positive")
        if int(online_kd.get("teacher_batch_size", 4)) <= 0:
            raise ValueError("online_kd.teacher_batch_size must be positive")
    for field in ("source_field", "target_field", "id_field"):
        if not str(data.get(field, field.removesuffix("_field"))).strip():
            raise ValueError(f"data.{field} cannot be empty")
    record_mapper = str(data.get("record_mapper", "")).strip()
    if record_mapper and ":" not in record_mapper:
        raise ValueError("data.record_mapper must use module:function syntax")

    checkpoint = config.get("checkpoint", {})
    if bool(checkpoint.get("save_best", False)):
        best_metric = str(checkpoint.get("best_metric", "eval_loss_ce"))
        if not best_metric.startswith("eval_"):
            raise ValueError("checkpoint.best_metric must name a validation metric beginning with eval_")
        if str(checkpoint.get("best_mode", "min")) not in {"min", "max"}:
            raise ValueError("checkpoint.best_mode must be min or max")
