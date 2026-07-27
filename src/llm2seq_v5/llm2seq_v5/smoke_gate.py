"""Fail-fast sanity gate for a completed v5 smoke run.

This gate detects broken generation and structurally disconnected source paths.
Learned path-strength diagnostics remain warnings until calibrated by matched
held-out pilots. It deliberately does not use ROUGE as a success criterion: a
100-example smoke run cannot establish generalization or beat T5Gemma.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _last_validation(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No validation entries in {path}")
    return rows[-1]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_smoke_run(run_dir: str | Path, expected_examples: int = 20) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    metrics = _read_json(run_dir / "last_validation_predictions.metrics.json")
    validation = _last_validation(run_dir / "validation_history.jsonl")
    required_validation = (
        "eval_loss",
        "eval_loss_ce",
        "eval_loss_response_alignment",
        "eval_response_alignment_cosine",
        "eval_loss_phrase_mixture",
        "eval_loss_phrase_copy",
        "eval_loss_phrase_continue",
        "eval_loss_phrase_labels",
        "eval_loss_phrase_coverage",
        "eval_phrase_mode_generate",
        "eval_phrase_mode_new",
        "eval_phrase_mode_continue",
        "eval_summary_prefix_rms",
        "eval_prefix_to_embedding_rms_ratio",
        "eval_loss_contrastive",
        "eval_loss_source_swap",
        "eval_cross_residual_ratio",
        "eval_prefix_swap_nll_gap",
        "eval_prefix_swap_accuracy",
    )
    routes = {
        name: float(validation.get(f"eval_memory_route_{name}", 0.0)) for name in ("lexical", "semantic", "summary")
    }
    route_metrics = metrics.get("memory_route_mean", {})
    inferred_bank_count = len(route_metrics) if isinstance(route_metrics, dict) and route_metrics else 1
    memory_bank_count = int(metrics.get("memory_bank_count", inferred_bank_count))
    multi_bank = memory_bank_count > 1
    training_parameters = int(metrics.get("training_parameters", 0))
    declared_parameter_target = int(metrics.get("parameter_target_declared", 0))
    # Non-blocking diagnostics collected alongside the hard gates. These are
    # reported rather than gated because their healthy bands have never been
    # measured on a completed run; promote to hard gates once a pilot establishes
    # them.
    warnings: list[str] = []
    gates = {
        "complete_checkpoint": (run_dir / "COMPLETE").is_file() and (run_dir / "last.pt").is_file(),
        "expected_heldout_examples": int(metrics.get("num_examples", -1)) == int(expected_examples),
        "validation_only_selection": metrics.get("evaluation_split") == "validation",
        "checkpoint_parameter_integrity": bool(metrics.get("checkpoint_parameters_match_model", False)),
        "below_t5gemma_parameter_budget": bool(metrics.get("parameter_budget_reached", False)),
        "total_below_declared_t5gemma_budget": (
            training_parameters > 0
            and declared_parameter_target > 0
            and training_parameters < declared_parameter_target
        ),
        "finite_diagnostic_rouge": all(_finite(metrics.get(name)) for name in ("rouge1", "rouge2", "rougeL")),
        "no_empty_generation": float(metrics.get("empty_prediction_rate", 100.0)) == 0.0,
        "nontrivial_length": float(metrics.get("prediction_words_mean", 0.0)) >= 5.0,
        "prediction_diversity": float(metrics.get("unique_prediction_rate", 0.0)) >= 70.0,
        "no_dominant_fixed_prefix": float(metrics.get("dominant_prefix_5gram_rate", 100.0)) <= 30.0,
        "limited_trigram_repetition": float(metrics.get("repeated_trigram_rate_mean", 100.0)) <= 20.0,
        "finite_source_diagnostics": all(_finite(validation.get(name)) for name in required_validation),
        # Flow check only. Magnitude does not establish useful source dependence;
        # the paired source/prefix interventions below are the causal diagnostics.
        "cross_attention_connected": float(validation.get("eval_cross_residual_ratio", 0.0)) > 1e-4,
        "summary_prefix_nonzero": float(validation.get("eval_summary_prefix_rms", 0.0)) > 1e-4,
    }
    if bool(metrics.get("phrase_pointer_enabled", False)):
        generated_modes = metrics.get("phrase_generation_modes", {})
        mode_values = [generated_modes.get(name) for name in ("generate", "new_span", "continue_span")]
        gates["phrase_pointer_observed_in_generation"] = (
            int(metrics.get("phrase_generation_mode_observations", 0)) > 0
            and all(_finite(value) for value in mode_values)
            and abs(sum(float(value) for value in mode_values) - 1.0) < 1e-4
        )
        if all(_finite(value) for value in mode_values) and float(generated_modes.get("generate", 1.0)) > 0.999:
            warnings.append(
                "phrase pointer remains almost entirely in generate mode; run the matched pilot "
                "before attributing any ROUGE change to phrase continuation"
            )
    if multi_bank:
        entropy = validation.get("eval_memory_routing_entropy")
        gates["finite_router_diagnostics"] = _finite(entropy) and all(_finite(value) for value in routes.values())
        if _finite(entropy) and (float(entropy) < 0.30 or min(routes.values()) <= 0.01):
            warnings.append(
                "multi-bank routing is highly concentrated; compare bank interventions before calling this collapse"
            )
        if _finite(entropy) and float(entropy) >= 0.995:
            warnings.append(
                "multi-bank routing is nearly uniform; entropy cannot tell whether the banks are "
                "complementary or duplicate, so inspect bank cosine/intervention diagnostics"
            )
    if bool(metrics.get("query_adaptive_routing", False)):
        adaptive_delta = validation.get("eval_adaptive_routing_delta")
        gates["finite_adaptive_router_diagnostic"] = _finite(adaptive_delta)
        if _finite(adaptive_delta) and float(adaptive_delta) <= 1e-7:
            warnings.append("query-adaptive router has not measurably departed from its static prior")
    if bool(metrics.get("source_swap", False)):
        swap_accuracy = float(validation.get("eval_source_swap_accuracy", 0.0))
        swap_gap = float(validation.get("eval_source_swap_nll_gap", 0.0))
        validation_examples = int(validation.get("eval_examples", 0))
        if swap_accuracy <= 0.60 or swap_gap <= 0.0:
            warnings.append(
                f"correct-source preference is weak on {validation_examples} validation examples "
                f"(accuracy={swap_accuracy:.3f}, NLL gap={swap_gap:.4f}); treat this as a pilot "
                "hypothesis, not a calibrated smoke failure"
            )
    failures = [name for name, passed in gates.items() if not passed]
    prefix_ratio = validation.get("eval_prefix_to_embedding_rms_ratio")
    prefix_drift = validation.get("eval_prefix_drift_ratio")
    # Prefix self-drift only describes transformation of prefix positions. It
    # cannot prove that target positions use those states through causal self-
    # attention, so report it without inferring "inert" versus "healthy".
    if _finite(prefix_drift) and float(prefix_drift) < 0.05:
        warnings.append(
            f"prefix-state cosine distance is {float(prefix_drift):.4f}; this is descriptive only. "
            "Use eval_prefix_swap_nll_gap to decide whether target predictions use the prefix"
        )
    if _finite(prefix_ratio) and not (0.3 < float(prefix_ratio) < 3.0):
        warnings.append(
            f"summary prefix RMS is {float(prefix_ratio):.2f}x the decoder token-embedding RMS "
            "(nominal target ~1x). This is a scale warning, not evidence that the prefix is unused"
        )
    prefix_swap_gap = validation.get("eval_prefix_swap_nll_gap")
    prefix_swap_accuracy = validation.get("eval_prefix_swap_accuracy")
    if (
        _finite(prefix_swap_gap)
        and _finite(prefix_swap_accuracy)
        and (float(prefix_swap_gap) <= 0.0 or float(prefix_swap_accuracy) <= 0.55)
    ):
        warnings.append(
            "swapping only the source-conditioned prefix does not reliably worsen target NLL "
            f"(gap={float(prefix_swap_gap):.4f}, accuracy={float(prefix_swap_accuracy):.3f}); "
            "the dense-memory path may make the prefix redundant"
        )
    return {
        "scope": "flow/collapse smoke gate only; not a generalization or T5Gemma comparison",
        # These values are a Unicode-preserving whitespace diagnostic. Vietnamese
        # whitespace is syllable-level rather than a fully segmented linguistic
        # word metric, and this backend is not on the Perl ROUGE scale.
        "rouge_backend_warning": (
            "rouge* below use rouge==1.0.0 (Unicode-preserving whitespace diagnostic). "
            "NOT on the same scale as the Perl ROUGE-1.5.5 benchmark targets in base.yaml; "
            "do not mix the two scales"
        ),
        "run_dir": str(run_dir),
        "memory_bank_count": memory_bank_count,
        "passed": not failures,
        "failed_gates": failures,
        "warnings": warnings,
        "gates": gates,
        "generation_diagnostics": {
            name: metrics.get(name)
            for name in (
                "num_examples",
                "rouge1",
                "rouge2",
                "rougeL",
                "prediction_words_mean",
                "empty_prediction_rate",
                "unique_prediction_rate",
                "dominant_prefix_5gram_rate",
                "repeated_trigram_rate_mean",
                "deployable_parameters",
                "training_parameters",
                "parameter_target_declared",
                "phrase_pointer_enabled",
                "phrase_generation_modes",
                "phrase_generation_mode_observations",
            )
        },
        "source_diagnostics": {
            name: validation.get(name)
            for name in (
                *required_validation,
                "eval_prompt_retrieval_accuracy",
                "eval_source_swap_accuracy",
                "eval_source_swap_nll_gap",
                "eval_prefix_swap_nll_gap",
                "eval_prefix_swap_accuracy",
                "eval_response_alignment_cosine",
                "eval_response_alignment_accuracy",
                "eval_summary_prefix_rms",
                "eval_memory_routing_entropy",
                "eval_adaptive_routing_delta",
                # Computed in model.py but previously dropped by the training
                # allow-list, so they never reached a run directory.
                "eval_salience_precision",
                "eval_salience_recall",
                "eval_projection_gate",
                "eval_salience_attention_gate",
            )
        },
        "mean_routes": routes if multi_bank else {"summary": 1.0},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-examples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_smoke_run(args.run_dir, args.expected_examples)
    output = args.output or args.run_dir / "smoke_gate.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
