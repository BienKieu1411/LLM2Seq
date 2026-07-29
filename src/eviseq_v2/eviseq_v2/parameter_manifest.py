"""Exact unique-parameter accounting for the EviSeq paper contract."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn


def _component(name: str) -> str:
    if name.startswith("alignment_head."):
        return "training_only_contrastive"
    if name.startswith("adapter."):
        return "bridge"
    if name.startswith("encoder."):
        if any(
            marker in name
            for marker in (
                "evidence_norm",
                "evidence_head",
                "evidence_view_gate",
                "generic_token_gate",
            )
        ):
            return "encoder_interface"
        return "encoder_backbone"
    if name.startswith("decoder."):
        if ".cross_attn" in name or name.endswith(".cross_gate"):
            return "decoder_cross_attention"
        return "decoder_backbone"
    raise ValueError(f"Unclassified EviSeq parameter: {name}")


def build_parameter_manifest(model: nn.Module, config: Dict[str, Any]) -> Dict[str, Any]:
    """Count unique parameter objects and enforce the conservative budget.

    The paper budget uses every parameter resident in the trained checkpoint,
    including the training-only InfoNCE projections.  Smaller deployment and
    actually executed counts are reported separately, never substituted for
    the conservative headline count.
    """

    named = list(model.named_parameters(remove_duplicate=True))
    if len({id(parameter) for _, parameter in named}) != len(named):
        raise RuntimeError("named_parameters(remove_duplicate=True) returned duplicate objects")

    by_component: Dict[str, int] = {}
    by_name = {name: int(parameter.numel()) for name, parameter in named}
    for name, count in by_name.items():
        component = _component(name)
        by_component[component] = by_component.get(component, 0) + count

    resident = sum(by_name.values())
    if resident != sum(by_component.values()):
        raise RuntimeError("Parameter component accounting does not sum to the resident total")

    training_only_prefixes = ("alignment_head.", "evidence_contrastive_head.")
    training_only = sum(
        count for name, count in by_name.items() if any(name.startswith(p) for p in training_only_prefixes)
    )
    deployable_resident = resident - training_only

    attention = config.get("native_attention", {})
    inactive_prefixes = list(training_only_prefixes)
    if str(attention.get("backend", "qwen_native")) == "qwen_native":
        variant = str(attention.get("variant", "evidence"))
        if variant == "evidence":
            inactive_prefixes.append("encoder.generic_token_gate.")
        elif variant in {"causal", "full"}:
            inactive_prefixes.extend(("encoder.evidence_view_gate", "encoder.generic_token_gate."))
    inactive = sum(
        count
        for name, count in by_name.items()
        if any(name == prefix or name.startswith(prefix) for prefix in inactive_prefixes)
    )
    inference_active = resident - inactive

    reporting = config.get("reporting", {})
    eligible = bool(reporting.get("parameter_efficiency_claim_eligible", False))
    target = int(reporting.get("target_total_footprint_approx", 0))
    if eligible and target <= 0:
        raise ValueError("An eligible parameter claim requires a positive target footprint")
    strictly_under_budget = resident < target if target > 0 else None
    if eligible and not strictly_under_budget:
        raise RuntimeError(f"Resident parameter total {resident:,} is not strictly below the target {target:,}")

    return {
        "counting_rule": "named_parameters(remove_duplicate=True)",
        "resident_training_total_unique": resident,
        "deployable_resident_without_train_aux": deployable_resident,
        "inference_active_unique": inference_active,
        "training_only_total_unique": training_only,
        "inactive_at_inference_total_unique": inactive,
        "by_component": dict(sorted(by_component.items())),
        "parameter_efficiency_claim_eligible": eligible,
        "target_total_footprint_approx": target,
        "enforced_count": "resident_training_total_unique",
        "strictly_under_budget": strictly_under_budget,
        "resident_margin_to_target": target - resident if target > 0 else None,
        "architecture_sha256": config.get("_meta", {}).get("architecture_sha256"),
    }
