"""Fail-fast sanity gate for a completed v4 smoke run.

This gate detects broken generation, a disconnected latent prefix, and
disconnected source attention. It deliberately does not use ROUGE as a success criterion: a
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
    metrics = _read_json(run_dir / "last_test_predictions.metrics.json")
    validation = _last_validation(run_dir / "validation_history.jsonl")
    required_validation = (
        "eval_loss",
        "eval_loss_ce",
        "eval_loss_response_alignment",
        "eval_response_alignment_cosine",
        "eval_summary_prefix_rms",
        "eval_loss_contrastive",
        "eval_loss_source_swap",
        "eval_cross_residual_ratio",
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
    gates = {
        "complete_checkpoint": (run_dir / "COMPLETE").is_file() and (run_dir / "last.pt").is_file(),
        "expected_heldout_examples": int(metrics.get("num_examples", -1)) == int(expected_examples),
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
        "cross_attention_active": float(validation.get("eval_cross_residual_ratio", 0.0)) > 1e-4,
        "summary_prefix_nonzero": float(validation.get("eval_summary_prefix_rms", 0.0)) > 1e-4,
    }
    if multi_bank:
        gates["router_not_collapsed"] = (
            float(validation.get("eval_memory_routing_entropy", 0.0)) > 0.30 and min(routes.values()) > 0.01
        )
    if bool(metrics.get("query_adaptive_routing", False)):
        gates["adaptive_router_updated"] = (
            _finite(validation.get("eval_adaptive_routing_delta"))
            and float(validation.get("eval_adaptive_routing_delta", 0.0)) > 1e-7
        )
    if bool(metrics.get("source_swap", False)):
        gates["correct_source_preferred"] = (
            float(validation.get("eval_source_swap_accuracy", 0.0)) > 0.55
            and float(validation.get("eval_source_swap_nll_gap", 0.0)) > 0.0
        )
    failures = [name for name, passed in gates.items() if not passed]
    return {
        "scope": "flow/collapse smoke gate only; not a generalization or T5Gemma comparison",
        "run_dir": str(run_dir),
        "memory_bank_count": memory_bank_count,
        "passed": not failures,
        "failed_gates": failures,
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
            )
        },
        "source_diagnostics": {
            name: validation.get(name)
            for name in (
                *required_validation,
                "eval_prompt_retrieval_accuracy",
                "eval_source_swap_accuracy",
                "eval_source_swap_nll_gap",
                "eval_response_alignment_cosine",
                "eval_response_alignment_accuracy",
                "eval_summary_prefix_rms",
                "eval_memory_routing_entropy",
                "eval_adaptive_routing_delta",
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
