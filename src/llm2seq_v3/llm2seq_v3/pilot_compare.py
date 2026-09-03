"""Compare held-out HiRoute and matched single-bank pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


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


def _same_data(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return bool(left and right) and int(left.get("num_examples", -1)) == int(right.get("num_examples", -2))


def compare_pilots(main_dir: Path, control_dir: Path) -> Dict[str, Any]:
    main_metrics = _read_json(main_dir / "last_test_predictions.metrics.json")
    control_metrics = _read_json(control_dir / "last_test_predictions.metrics.json")
    main_validation = _last_jsonl(main_dir / "validation_history.jsonl")
    control_validation = _last_jsonl(control_dir / "validation_history.jsonl")
    main_test = main_metrics.get("test_data_record", {})
    control_test = control_metrics.get("test_data_record", {})
    comparability_gates = {
        "same_test_size": int(main_metrics.get("num_examples", -1)) == int(control_metrics.get("num_examples", -2)),
        "same_test_data": _same_data(main_test, control_test),
        "main_checkpoint_integrity": bool(main_metrics.get("checkpoint_parameters_match_model", False)),
        "control_checkpoint_integrity": bool(control_metrics.get("checkpoint_parameters_match_model", False)),
        "same_encoder": str(main_metrics.get("encoder_name", ""))
        == str(control_metrics.get("encoder_name", "missing")),
        "same_decoder": str(main_metrics.get("decoder_name", ""))
        == str(control_metrics.get("decoder_name", "missing")),
        "same_validation_scope": (
            int(main_validation.get("epoch", -1)) == int(control_validation.get("epoch", -2))
            and int(main_validation.get("eval_examples", -1)) == int(control_validation.get("eval_examples", -2))
        ),
        "same_diagnostic_rouge_backend": (
            "rouge==1.0.0" in str(main_metrics.get("rouge_backend", ""))
            and str(main_metrics.get("rouge_backend")) == str(control_metrics.get("rouge_backend"))
        ),
    }
    comparable = all(comparability_gates.values())
    rouge_delta = {
        name: round(float(main_metrics[name]) - float(control_metrics[name]), 4)
        for name in ("rouge1", "rouge2", "rougeL")
    }
    retrieval = float(main_validation.get("eval_prompt_retrieval_accuracy", 0.0))
    swap_accuracy = float(main_validation.get("eval_source_swap_accuracy", 0.0))
    swap_gap = float(main_validation.get("eval_source_swap_nll_gap", 0.0))
    negative_similarity = float(main_validation.get("eval_source_swap_negative_similarity", 0.0))
    routing_entropy = float(main_validation.get("eval_memory_routing_entropy", 0.0))
    adaptive_delta = float(main_validation.get("eval_adaptive_routing_delta", 0.0))
    routes = {
        name: float(main_validation.get(f"eval_memory_route_{name}", 0.0))
        for name in ("lexical", "semantic", "summary")
    }
    gates = {
        "hiroute_improves_rouge2": rouge_delta["rouge2"] > 0.0,
        "hiroute_does_not_regress_rouge1_or_l": (rouge_delta["rouge1"] >= -0.5 and rouge_delta["rougeL"] >= -0.5),
        "prompt_retrieval_above_5_percent": retrieval > 0.05,
        "correct_source_preferred": swap_accuracy > 0.55 and swap_gap > 0.0,
        "router_not_collapsed": routing_entropy > 0.60 and min(routes.values()) > 0.05,
        "adaptive_router_updated": adaptive_delta > 1e-7,
    }
    return {
        "comparison_scope": "held-out diagnostic pilot; not a paper score",
        "num_test_examples": int(main_metrics["num_examples"]),
        "comparable": comparable,
        "comparability_gates": comparability_gates,
        "main": {name: float(main_metrics[name]) for name in ("rouge1", "rouge2", "rougeL")},
        "single_bank_control": {name: float(control_metrics[name]) for name in ("rouge1", "rouge2", "rougeL")},
        "hiroute_minus_control": rouge_delta,
        "main_source_utilization": {
            "prompt_retrieval_accuracy": retrieval,
            "source_swap_accuracy": swap_accuracy,
            "source_swap_nll_gap": swap_gap,
            "source_swap_negative_similarity": negative_similarity,
            "routing_entropy": routing_entropy,
            "adaptive_routing_delta": adaptive_delta,
            "mean_routes": routes,
        },
        "gates": gates,
        "recommend_full_hiroute_run": comparable and all(gates.values()),
        "main_validation": main_validation,
        "control_validation": control_validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_pilots(args.main_dir, args.control_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
