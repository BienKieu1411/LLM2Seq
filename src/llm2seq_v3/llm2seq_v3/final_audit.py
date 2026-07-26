"""Strict artifact-to-artifact audit for the final T5Gemma paper claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .config import load_config

_ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")
_BASELINE_SCORE_TOLERANCE = 0.02


def _scores(payload: Dict[str, Any], label: str) -> Dict[str, float]:
    missing = [key for key in _ROUGE_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{label} score artifact is missing {missing}")
    return {key: float(payload[key]) for key in _ROUGE_KEYS}


def _same_fingerprint(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return bool(left and right) and (
        int(left.get("num_examples", -1)) == int(right.get("num_examples", -2))
        and str(left.get("sha256", "")) == str(right.get("sha256", "missing"))
    )


def _normalized_model_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def audit_final_claim(
    config: Dict[str, Any],
    candidate_scores_payload: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    baseline_scores_payload: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Prove comparability, smaller size, and strict ROUGE superiority.

    This deliberately does not use the rounded ``2B`` model-card size for the
    final parameter claim. It compares the unique runtime parameter elements
    reported by the two instantiated checkpoints.
    """

    paper_target = config.get("benchmark", {}).get("paper", {})
    locked_test = config.get("benchmark", {}).get("data", {}).get("test", {})
    expected_examples = int(paper_target.get("num_examples", 0))
    if expected_examples <= 0 or not locked_test:
        raise ValueError("Config has no locked paper test split")

    candidate_scores = _scores(candidate_scores_payload, "candidate")
    baseline_scores = _scores(baseline_scores_payload, "baseline")
    locked_scores = _scores(paper_target, "configured T5Gemma baseline")
    comparability_reasons: list[str] = []

    for label, payload in (
        ("candidate", candidate_scores_payload),
        ("baseline", baseline_scores_payload),
    ):
        if "Perl ROUGE-1.5.5" not in str(payload.get("backend", "")):
            comparability_reasons.append(f"{label} backend is not Perl ROUGE-1.5.5")
        actual_examples = int(payload.get("num_examples", 0))
        if actual_examples != expected_examples:
            comparability_reasons.append(f"{label} has {actual_examples} examples; locked test has {expected_examples}")

    candidate_test = candidate_metrics.get("test_data_fingerprint", {})
    baseline_test = baseline_metrics.get("test_data_fingerprint", {})
    for label, fingerprint in (
        ("candidate", candidate_test),
        ("baseline", baseline_test),
    ):
        if not _same_fingerprint(fingerprint, locked_test):
            comparability_reasons.append(f"{label} test fingerprint does not match the locked T5Gemma split")
    if not _same_fingerprint(candidate_test, baseline_test):
        comparability_reasons.append("candidate and baseline were not evaluated on the same test rows")

    if not bool(candidate_metrics.get("checkpoint_test_matches_current", False)):
        comparability_reasons.append("candidate checkpoint test fingerprint does not match its evaluated file")
    if not bool(candidate_metrics.get("checkpoint_parameters_match_model", False)):
        comparability_reasons.append("candidate checkpoint parameter count does not match its instantiated model")
    if not bool(baseline_metrics.get("checkpoint_test_matches_current", False)):
        comparability_reasons.append("baseline checkpoint test fingerprint does not match its evaluated file")
    if baseline_metrics.get("checkpoint_parameters_match_model") is False:
        comparability_reasons.append("baseline checkpoint parameter count does not match its instantiated model")

    expected_model_id = _normalized_model_name(paper_target.get("model_id", "google/t5gemma-2-1b-1b"))
    expected_model_slug = expected_model_id.rsplit("/", maxsplit=1)[-1]
    baseline_identities = {
        _normalized_model_name(baseline_metrics.get("base_model")),
        _normalized_model_name(baseline_metrics.get("checkpoint_base_model")),
    }
    if not any(expected_model_slug in value for value in baseline_identities):
        comparability_reasons.append(f"baseline identity is not the locked model {paper_target.get('model_id')}")

    baseline_matches_locked_scores = all(
        abs(baseline_scores[key] - locked_scores[key]) <= _BASELINE_SCORE_TOLERANCE for key in _ROUGE_KEYS
    )
    if not baseline_matches_locked_scores:
        comparability_reasons.append("baseline Perl ROUGE artifact does not reproduce the locked T5Gemma scores")

    candidate_deployable_parameters = int(candidate_metrics.get("deployable_parameters", 0))
    candidate_total_parameters = int(candidate_metrics.get("training_parameters", 0))
    baseline_parameters = int(
        baseline_metrics.get(
            "unique_parameter_elements",
            baseline_metrics.get("total_parameters", 0),
        )
    )
    if candidate_deployable_parameters <= 0:
        comparability_reasons.append("candidate deployable parameter count is missing")
    if candidate_total_parameters <= 0:
        comparability_reasons.append("candidate total parameter count is missing")
    if (
        candidate_deployable_parameters > 0
        and candidate_total_parameters > 0
        and candidate_deployable_parameters > candidate_total_parameters
    ):
        comparability_reasons.append("candidate deployable count exceeds its total count")
    if baseline_parameters <= 0:
        comparability_reasons.append("baseline exact runtime parameter count is missing")

    comparable = not comparability_reasons
    total_parameter_delta = candidate_total_parameters - baseline_parameters
    deployable_parameter_delta = candidate_deployable_parameters - baseline_parameters
    total_smaller_than_baseline = (
        candidate_total_parameters > 0 and baseline_parameters > 0 and candidate_total_parameters < baseline_parameters
    )
    deployable_smaller_than_baseline = (
        candidate_deployable_parameters > 0
        and baseline_parameters > 0
        and candidate_deployable_parameters < baseline_parameters
    )
    smaller_than_baseline = total_smaller_than_baseline and deployable_smaller_than_baseline
    deltas = {key: round(candidate_scores[key] - baseline_scores[key], 4) for key in _ROUGE_KEYS}
    strictly_surpassed = {key: candidate_scores[key] > baseline_scores[key] for key in _ROUGE_KEYS}
    all_rouge_strictly_surpassed = all(strictly_surpassed.values())

    claim_failures: list[str] = []
    if comparable and not smaller_than_baseline:
        claim_failures.append(
            "candidate total and deployable counts are not both strictly smaller than the actual T5Gemma checkpoint"
        )
    if comparable and not all_rouge_strictly_surpassed:
        missed = [key for key, passed in strictly_surpassed.items() if not passed]
        claim_failures.append(f"candidate does not strictly surpass T5Gemma on {missed}")

    passed = bool(comparable and smaller_than_baseline and all_rouge_strictly_surpassed)
    return {
        "passed": passed,
        "claim": ("LLM2Seq-v3 strictly exceeds T5Gemma2-1B-1B with fewer total and deployable parameters"),
        "comparable": comparable,
        "comparability_reasons": comparability_reasons,
        "claim_failures": claim_failures,
        "candidate": {
            "scores": candidate_scores,
            "total_parameters": candidate_total_parameters,
            "deployable_parameters": candidate_deployable_parameters,
            "test_fingerprint": candidate_test,
        },
        "baseline": {
            "name": paper_target.get("name"),
            "model_id": paper_target.get("model_id"),
            "scores": baseline_scores,
            "unique_parameter_elements": baseline_parameters,
            "test_fingerprint": baseline_test,
            "matches_locked_scores": baseline_matches_locked_scores,
        },
        "candidate_minus_baseline": deltas,
        "candidate_total_minus_baseline_parameters": total_parameter_delta,
        "candidate_deployable_minus_baseline_parameters": deployable_parameter_delta,
        "total_parameter_reduction_percent": (
            round(
                100.0 * (baseline_parameters - candidate_total_parameters) / baseline_parameters,
                4,
            )
            if baseline_parameters > 0 and candidate_total_parameters > 0
            else None
        ),
        "strictly_surpassed": strictly_surpassed,
        "all_rouge_strictly_surpassed": all_rouge_strictly_surpassed,
        "smaller_than_baseline": smaller_than_baseline,
        "total_smaller_than_baseline": total_smaller_than_baseline,
        "deployable_smaller_than_baseline": deployable_smaller_than_baseline,
        "parameter_comparison_uses_actual_runtime_counts": True,
        "rounded_2b_count_used_for_final_claim": False,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly audit V3 against the actual T5Gemma2-1B-1B artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-scores", required=True, type=Path)
    parser.add_argument("--candidate-metrics", required=True, type=Path)
    parser.add_argument("--baseline-scores", required=True, type=Path)
    parser.add_argument("--baseline-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = audit_final_claim(
        load_config(args.config),
        _read_json(args.candidate_scores),
        _read_json(args.candidate_metrics),
        _read_json(args.baseline_scores),
        _read_json(args.baseline_metrics),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
