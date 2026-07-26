"""Compare held-out encoder-system pilots without treating them as paper scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _last_jsonl(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No validation rows in {path}")
    return rows[-1]


def _fingerprint_key(metrics: Mapping[str, Any]) -> tuple[int, str]:
    fingerprint = metrics.get("test_data_fingerprint", {})
    return int(fingerprint.get("num_examples", -1)), str(fingerprint.get("sha256", ""))


def compare_encoder_pilots(run_dirs: Mapping[str, Path]) -> Dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("At least two encoder pilot runs are required")
    records: Dict[str, Dict[str, Any]] = {}
    for label, run_dir in run_dirs.items():
        metrics = _read_json(run_dir / "last_test_predictions.metrics.json")
        validation = _last_jsonl(run_dir / "validation_history.jsonl")
        routes = {
            name: float(validation.get(f"eval_memory_route_{name}", 0.0)) for name in ("lexical", "semantic", "summary")
        }
        route_metrics = metrics.get("memory_route_mean", {})
        inferred_bank_count = len(route_metrics) if isinstance(route_metrics, dict) and route_metrics else 1
        memory_bank_count = int(metrics.get("memory_bank_count", inferred_bank_count))
        multi_bank = memory_bank_count > 1
        training_parameters = int(metrics.get("training_parameters", 0))
        parameter_target = int(metrics.get("parameter_target_declared", 0))
        healthy = {
            "checkpoint_integrity": bool(metrics.get("checkpoint_parameters_match_model", False)),
            "parameter_budget": bool(metrics.get("parameter_budget_reached", False)),
            "total_parameter_budget": (
                training_parameters > 0 and parameter_target > 0 and training_parameters < parameter_target
            ),
            "nonempty_generation": float(metrics.get("empty_prediction_rate", 100.0)) == 0.0,
            "cross_attention_active": float(validation.get("eval_cross_residual_ratio", 0.0)) > 1e-4,
            "correct_source_preferred": (
                float(validation.get("eval_source_swap_accuracy", 0.0)) > 0.55
                and float(validation.get("eval_source_swap_nll_gap", 0.0)) > 0.0
            ),
        }
        if multi_bank:
            healthy["router_not_collapsed"] = (
                float(validation.get("eval_memory_routing_entropy", 0.0)) > 0.60 and min(routes.values()) > 0.05
            )
        records[label] = {
            "run_dir": str(run_dir),
            "encoder_name": str(metrics.get("encoder_name", "")),
            "decoder_name": str(metrics.get("decoder_name", "")),
            "rouge1": float(metrics["rouge1"]),
            "rouge2": float(metrics["rouge2"]),
            "rougeL": float(metrics["rougeL"]),
            "num_examples": int(metrics.get("num_examples", -1)),
            "test_data_fingerprint": metrics.get("test_data_fingerprint", {}),
            "rouge_backend": str(metrics.get("rouge_backend", "")),
            "deployable_parameters": int(metrics.get("deployable_parameters", 0)),
            "training_parameters": training_parameters,
            "parameter_target_declared": parameter_target,
            "memory_bank_count": memory_bank_count,
            "validation_epoch": int(validation.get("epoch", -1)),
            "validation_examples": int(validation.get("eval_examples", -1)),
            "prompt_retrieval_accuracy": float(validation.get("eval_prompt_retrieval_accuracy", 0.0)),
            "source_swap_accuracy": float(validation.get("eval_source_swap_accuracy", 0.0)),
            "source_swap_nll_gap": float(validation.get("eval_source_swap_nll_gap", 0.0)),
            "routing_entropy": float(validation.get("eval_memory_routing_entropy", 0.0)),
            "mean_routes": routes,
            "health_gates": healthy,
            "healthy": all(healthy.values()),
        }

    values = list(records.values())
    reference = values[0]
    comparability_gates = {
        "same_test_fingerprint": all(_fingerprint_key(record) == _fingerprint_key(reference) for record in values[1:]),
        "same_test_size": all(record["num_examples"] == reference["num_examples"] for record in values[1:]),
        "same_decoder": all(record["decoder_name"] == reference["decoder_name"] for record in values[1:]),
        "same_validation_scope": all(
            record["validation_epoch"] == reference["validation_epoch"]
            and record["validation_examples"] == reference["validation_examples"]
            for record in values[1:]
        ),
        "same_diagnostic_rouge_backend": all(
            record["rouge_backend"] == reference["rouge_backend"] and "rouge==1.0.0" in record["rouge_backend"]
            for record in values
        ),
        "distinct_encoders": len({record["encoder_name"] for record in values}) == len(values),
    }
    comparable = all(comparability_gates.values())
    eligible = [(label, record) for label, record in records.items() if record["healthy"]]
    ranking = [
        label
        for label, _ in sorted(
            eligible,
            key=lambda item: (
                item[1]["rouge2"],
                item[1]["rouge1"],
                item[1]["rougeL"],
            ),
            reverse=True,
        )
    ]
    recommendation = ranking[0] if comparable and ranking else None
    return {
        "comparison_scope": (
            "held-out encoder-system pilot; PPLX is a shape-matched encoder swap, while the "
            "Nemotron profile is budget-matched with sparser cross-attention; not a paper score"
        ),
        "comparable": comparable,
        "comparability_gates": comparability_gates,
        "ranking_by_rouge2_then_rouge1_rougeL": ranking,
        "recommended_for_full_run": recommendation,
        "runs": records,
    }


def _parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run must be LABEL=RUN_DIR")
    return label.strip(), Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=_parse_run)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_dirs = dict(args.run)
    if len(run_dirs) != len(args.run):
        raise ValueError("Encoder pilot labels must be unique")
    result = compare_encoder_pilots(run_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
