"""Compare only like-for-like Perl ROUGE-1.5.5 paper scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .config import load_config


def compare_paper_scores(
    config: Dict[str, Any],
    scores: Dict[str, Any],
    candidate_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    target = config.get("benchmark", {}).get("paper", {})
    required = ("rouge1", "rouge2", "rougeL")
    if not all(name in target for name in required):
        raise ValueError("Config has no complete benchmark.paper ROUGE target")
    backend = str(scores.get("backend", ""))
    expected_examples = int(target.get("num_examples", 0))
    actual_examples = int(scores.get("num_examples", 0))
    reasons = []
    if "Perl ROUGE-1.5.5" not in backend:
        reasons.append("candidate backend is not Perl ROUGE-1.5.5")
    if expected_examples > 0 and actual_examples != expected_examples:
        reasons.append(f"candidate has {actual_examples} examples; paper target uses {expected_examples}")
    parameter_target = config.get("benchmark", {}).get("parameters", {})
    locked_test = config.get("benchmark", {}).get("data", {}).get("test", {})
    if candidate_metrics is None:
        reasons.append("candidate diagnostic metrics are missing")
        deployable_parameters = 0
        training_parameters = 0
        current_test = {}
    else:
        deployable_parameters = int(candidate_metrics.get("deployable_parameters", 0))
        training_parameters = int(candidate_metrics.get("training_parameters", 0))
        current_test = candidate_metrics.get("test_data_fingerprint", {})
        if not bool(candidate_metrics.get("checkpoint_test_matches_current", False)):
            reasons.append("checkpoint test fingerprint does not match the evaluated test file")
        if not bool(candidate_metrics.get("checkpoint_parameters_match_model", False)):
            reasons.append("checkpoint parameter count does not match the instantiated candidate")
        if int(current_test.get("num_examples", -1)) != int(locked_test.get("num_examples", -2)):
            reasons.append("candidate test-set size does not match the locked test manifest")
        if str(current_test.get("sha256", "")) != str(locked_test.get("sha256", "missing")):
            reasons.append("candidate test fingerprint does not match the T5Gemma paper split")
    target_parameters = int(parameter_target.get("target_declared_parameters", 0))
    parameter_budget_reached = (
        deployable_parameters > 0 and target_parameters > 0 and deployable_parameters < target_parameters
    )
    if not parameter_budget_reached:
        reasons.append("candidate deployable parameter count is missing or not below the target")
    result: Dict[str, Any] = {
        "comparable": not reasons,
        "reasons": reasons,
        "candidate_backend": backend,
        "target_backend": str(target.get("backend", "Perl ROUGE-1.5.5")),
        "candidate_num_examples": actual_examples,
        "target_num_examples": expected_examples,
        "target_name": str(target.get("name", "T5Gemma")),
        "candidate_deployable_parameters": deployable_parameters,
        "candidate_training_parameters": training_parameters,
        "target_declared_parameters": target_parameters,
        "target_parameter_count_is_rounded": bool(parameter_target.get("target_is_rounded", False)),
        "parameter_budget_reached": parameter_budget_reached,
        "candidate_test_fingerprint": current_test,
        "target_test_fingerprint": locked_test,
        "candidate": {name: float(scores[name]) for name in required},
        "target": {name: float(target[name]) for name in required},
    }
    if result["comparable"]:
        result["candidate_minus_target"] = {
            name: round(result["candidate"][name] - result["target"][name], 4) for name in required
        }
        result["rouge2_target_reached"] = result["candidate"]["rouge2"] >= result["target"]["rouge2"]
        result["all_rouge_targets_reached"] = all(
            result["candidate"][name] >= result["target"][name] for name in required
        )
        result["paper_target_reached"] = bool(
            result["all_rouge_targets_reached"] and result["parameter_budget_reached"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-metrics", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    candidate_metrics = (
        json.loads(args.candidate_metrics.read_text(encoding="utf-8")) if args.candidate_metrics is not None else None
    )
    result = compare_paper_scores(config, scores, candidate_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
