"""Configuration loading and validation for LLM2Seq-v3."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("_base_", None)
    if parent:
        config = _merge(load_config(path.parent / str(parent)), config)
    validate_config(config)
    config.setdefault("_meta", {})["config_path"] = str(path)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    model = config.get("model", {})
    adapter = config.get("adapter", {})
    decoder = config.get("decoder", {})
    training = config.get("training", {})
    data = config.get("data", {})
    objectives = config.get("objectives", {})

    for field in ("encoder_name", "decoder_name"):
        if not str(model.get(field, "")).strip():
            raise ValueError(f"model.{field} is required")
    fallback_hidden = int(model.get("hidden_size", 0))
    encoder_hidden = int(model.get("encoder_hidden_size", fallback_hidden))
    decoder_hidden = int(model.get("decoder_hidden_size", fallback_hidden))
    if encoder_hidden <= 0 or decoder_hidden <= 0:
        raise ValueError("model encoder/decoder hidden sizes must be positive")
    encoder_layers = int(model.get("encoder_num_hidden_layers", 0))
    if encoder_layers < 0:
        raise ValueError("model.encoder_num_hidden_layers must be non-negative")
    attention_mode = str(model.get("encoder_attention_mode", "auto"))
    if attention_mode not in {"auto", "causal", "bidirectional"}:
        raise ValueError("model.encoder_attention_mode must be auto, causal, or bidirectional")
    if not str(model.get("encoder_attn_implementation", "sdpa")).strip():
        raise ValueError("model.encoder_attn_implementation cannot be empty")
    if "encoder_trust_remote_code" in model and not isinstance(model["encoder_trust_remote_code"], bool):
        raise ValueError("model.encoder_trust_remote_code must be a boolean")
    if "encoder_revision" in model and model["encoder_revision"] is not None:
        if not str(model["encoder_revision"]).strip():
            raise ValueError("model.encoder_revision cannot be empty")
    if float(model.get("encoder_future_context_min_relative_change", 1e-6)) <= 0.0:
        raise ValueError("model.encoder_future_context_min_relative_change must be positive")
    for field in ("append_source_eos", "source_add_special_tokens"):
        if field in data and not isinstance(data[field], bool):
            raise ValueError(f"data.{field} must be a boolean")
    if bool(data.get("append_source_eos", True)) and bool(data.get("source_add_special_tokens", False)):
        raise ValueError("data.append_source_eos and data.source_add_special_tokens cannot both be enabled")

    indices = list(adapter.get("fuse_layers", []))
    if bool(adapter.get("layer_fusion", True)) and not indices:
        raise ValueError("adapter.fuse_layers cannot be empty when layer fusion is enabled")
    if encoder_layers > 0:
        hidden_state_count = encoder_layers + 1
        indexed_fields = ["fuse_layers"]
        if bool(adapter.get("depth_routed_memory", False)):
            indexed_fields.extend(["lexical_layers", "semantic_layers"])
        for field in indexed_fields:
            for value in adapter.get(field, []):
                index = int(value)
                actual = index if index >= 0 else hidden_state_count + index
                if actual < 0 or actual >= hidden_state_count:
                    raise ValueError(
                        f"adapter.{field} index {index} is outside the encoder's {hidden_state_count} hidden states"
                    )
    if int(adapter.get("num_bidirectional_layers", 0)) < 0:
        raise ValueError("adapter.num_bidirectional_layers must be non-negative")
    hidden_size = int(adapter.get("hidden_size", decoder_hidden))
    heads = int(adapter.get("num_heads", 1))
    if hidden_size % heads:
        raise ValueError("adapter hidden size must be divisible by adapter.num_heads")
    depth_routed = bool(adapter.get("depth_routed_memory", False))
    expected_banks = 3 if depth_routed else 1
    configured_banks = int(decoder.get("memory_bank_count", 1))
    if configured_banks != expected_banks:
        raise ValueError(
            f"decoder.memory_bank_count must be {expected_banks} when adapter.depth_routed_memory={depth_routed}"
        )
    if depth_routed:
        for field in ("lexical_layers", "semantic_layers"):
            if not list(adapter.get(field, [])):
                raise ValueError(f"adapter.{field} cannot be empty for depth-routed memory")
        branch_context_gate = float(adapter.get("branch_context_gate_init", 0.1))
        if not 0.0 <= branch_context_gate < 1.0:
            raise ValueError("adapter.branch_context_gate_init must be in [0, 1)")
    if bool(adapter.get("hierarchical_sentence_context", False)):
        context_size = int(adapter.get("sentence_context_size", 256))
        context_heads = int(adapter.get("sentence_context_heads", 8))
        if context_size <= 0 or context_size % context_heads:
            raise ValueError(
                "adapter.sentence_context_size must be positive and divisible by adapter.sentence_context_heads"
            )

    cross_every = int(decoder.get("cross_attention_every", 1))
    if cross_every <= 0:
        raise ValueError("decoder.cross_attention_every must be positive")
    gate = float(decoder.get("cross_gate_init", 0.1))
    if not 0.0 <= gate < 1.0:
        raise ValueError("decoder.cross_gate_init must be in [0, 1)")
    routing_mode = str(decoder.get("memory_routing_mode", "attention_output"))
    if routing_mode not in {"attention_output", "memory"}:
        raise ValueError("decoder.memory_routing_mode must be 'attention_output' or 'memory'")
    query_adaptive_routing = bool(decoder.get("query_adaptive_routing", False))
    if query_adaptive_routing and (not depth_routed or routing_mode != "attention_output"):
        raise ValueError("decoder.query_adaptive_routing requires depth-routed memory and attention_output routing")
    if float(decoder.get("query_router_max_delta", 2.0)) <= 0.0:
        raise ValueError("decoder.query_router_max_delta must be positive")

    routing_balance_weight = float(objectives.get("routing_balance_weight", 0.0))
    if routing_balance_weight < 0.0:
        raise ValueError("objectives.routing_balance_weight must be non-negative")
    if routing_balance_weight > 0.0 and not depth_routed:
        raise ValueError("objectives.routing_balance_weight requires depth-routed memory")

    # --- Contrastive / objectives validation ---
    label_smoothing = float(objectives.get("label_smoothing", 0.1))
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("objectives.label_smoothing must be in [0, 1)")

    if bool(objectives.get("use_contrastive", True)):
        if not bool(objectives.get("use_prompt_alignment", True)) and not bool(objectives.get("use_source_swap", True)):
            raise ValueError(
                "At least one of objectives.use_prompt_alignment/use_source_swap must be true "
                "when contrastive learning is enabled"
            )
        contrastive_weight = float(objectives.get("contrastive_weight", 0.1))
        if contrastive_weight < 0.0:
            raise ValueError("objectives.contrastive_weight must be non-negative")
        temperature = float(objectives.get("contrastive_temperature", 0.07))
        if temperature <= 0.0:
            raise ValueError("objectives.contrastive_temperature must be positive")
        projection_size = int(objectives.get("contrastive_projection_size", 256))
        if projection_size <= 0:
            raise ValueError("objectives.contrastive_projection_size must be positive")
        contrastive_pooling = str(objectives.get("contrastive_pooling", "mean_last"))
        if contrastive_pooling not in {"mean", "mean_last"}:
            raise ValueError("objectives.contrastive_pooling must be 'mean' or 'mean_last'")
        contrastive_warmup = int(objectives.get("contrastive_warmup_epochs", 0))
        if contrastive_warmup < 0:
            raise ValueError("objectives.contrastive_warmup_epochs must be non-negative")
        source_swap_weight = float(objectives.get("source_swap_weight", 0.1))
        if source_swap_weight < 0.0:
            raise ValueError("objectives.source_swap_weight must be non-negative")
        source_swap_margin = float(objectives.get("source_swap_margin", 0.2))
        if source_swap_margin < 0.0:
            raise ValueError("objectives.source_swap_margin must be non-negative")
        source_swap_temperature = float(objectives.get("source_swap_temperature", 1.0))
        if source_swap_temperature <= 0.0:
            raise ValueError("objectives.source_swap_temperature must be positive")
        source_swap_strategy = str(objectives.get("source_swap_strategy", "hard_in_batch"))
        if source_swap_strategy not in {"hard_in_batch", "cyclic"}:
            raise ValueError("objectives.source_swap_strategy must be 'hard_in_batch' or 'cyclic'")

    warmup_epochs = int(training.get("interface_warmup_epochs", 0))
    full_epochs = int(training.get("full_finetune_epochs", 0))
    if warmup_epochs < 0 or full_epochs <= 0:
        raise ValueError("training requires non-negative warmup and positive full-finetune epochs")
    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive")
    if int(training.get("gradient_accumulation_steps", 0)) <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")

    for field in ("train_file", "validation_file", "test_file"):
        if not str(data.get(field, "")).strip():
            raise ValueError(f"data.{field} is required")
    if int(data.get("max_source_length", 0)) <= 0:
        raise ValueError("data.max_source_length must be positive")
    if int(data.get("max_target_length", 0)) <= 1:
        raise ValueError("data.max_target_length must be greater than one")

    benchmark = config.get("benchmark", {})
    locked_data = benchmark.get("data", {})
    for split in ("train", "validation", "test"):
        target = locked_data.get(split, {})
        if not target:
            continue
        if int(target.get("num_examples", 0)) <= 0:
            raise ValueError(f"benchmark.data.{split}.num_examples must be positive")
        digest = str(target.get("sha256", ""))
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest.lower()):
            raise ValueError(f"benchmark.data.{split}.sha256 must be a SHA-256 hex digest")
    parameter_target = benchmark.get("parameters", {})
    if parameter_target:
        if int(parameter_target.get("target_declared_parameters", 0)) <= 0:
            raise ValueError("benchmark.parameters.target_declared_parameters must be positive")
        if not str(parameter_target.get("target_name", "")).strip():
            raise ValueError("benchmark.parameters.target_name is required")
    for profile, required_backend in (
        ("diagnostic", "rouge==1.0.0"),
        ("paper", "Perl ROUGE-1.5.5"),
    ):
        target = benchmark.get(profile, {})
        if not target:
            continue
        if required_backend not in str(target.get("backend", "")):
            raise ValueError(
                f"benchmark.{profile}.backend must identify {required_backend}; "
                "cross-backend ROUGE comparisons are invalid"
            )
        if any(float(target.get(name, -1.0)) < 0.0 for name in ("rouge1", "rouge2", "rougeL")):
            raise ValueError(f"benchmark.{profile} requires non-negative ROUGE-1/2/L")
        if int(target.get("num_examples", 0)) <= 0:
            raise ValueError(f"benchmark.{profile}.num_examples must be positive")

    checkpoint = config.get("checkpoint", {})
    if bool(checkpoint.get("save_best", False)):
        raise ValueError("LLM2Seq-v3 intentionally supports last.pt only; set checkpoint.save_best=false")
    if bool(checkpoint.get("save_each_epoch", False)):
        raise ValueError("LLM2Seq-v3 intentionally avoids epoch checkpoints")
