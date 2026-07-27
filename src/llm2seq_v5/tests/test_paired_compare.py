from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import pytest
import yaml
from llm2seq_v5.metrics import rouge_scores
from llm2seq_v5.paired_compare import compare_paired_artifacts

REFERENCES = (
    "khởi động lại thiết bị và kiểm tra nguồn",
    "mở cài đặt rồi chọn mạng không dây",
    "rửa rau sạch và cắt thành miếng nhỏ",
    "đun nước sau đó cho trà vào ấm",
)
GENERATION = {
    "max_new_tokens": 256,
    "min_new_tokens": 16,
    "num_beams": 1,
    "do_sample": False,
    "repetition_penalty": 1.05,
    "no_repeat_ngram_size": 3,
}
SOURCE = {
    "source_prefix": "Tóm tắt:\n",
    "max_source_length": 3072,
    "max_target_length": 384,
}
FINGERPRINT = {"num_examples": 4, "sha256": "a" * 64}


def _write_jsonl(path: Path, predictions: Sequence[str], references: Sequence[str] = REFERENCES) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, (prediction, reference) in enumerate(zip(predictions, references)):
            handle.write(
                json.dumps(
                    {"id": f"test_{index:03d}", "prediction": prediction, "reference": reference},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_side(
    root: Path,
    label: str,
    predictions: Sequence[str],
    *,
    model_name: str,
    fingerprint: Dict[str, Any] = FINGERPRINT,
) -> Dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    predictions_path = root / f"{label}.jsonl"
    metrics_path = root / f"{label}.metrics.json"
    config_path = root / f"{label}.resolved.yaml"
    _write_jsonl(predictions_path, predictions)
    scores = rouge_scores(predictions, REFERENCES)
    metrics = {
        **scores,
        "num_examples": len(predictions),
        "test_data_fingerprint": fingerprint,
        "rouge_backend": "rouge==1.0.0 (HeterSumGraph)",
        "rouge_preprocessing": "NFC + lowercase + stored whitespace tokenization",
        "generation": dict(GENERATION),
        **SOURCE,
        "base_model": model_name,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "model": {"model_name_or_path": model_name},
        "data": dict(SOURCE),
        "generation": dict(GENERATION),
        "benchmark": {"data": {"test": fingerprint}},
        "limits": {"max_test_examples": 0},
    }
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"predictions": predictions_path, "metrics": metrics_path, "config": config_path}


def _compare(candidate: Dict[str, Path], baseline: Dict[str, Path]) -> Dict[str, Any]:
    return compare_paired_artifacts(
        candidate["predictions"],
        candidate["metrics"],
        candidate["config"],
        baseline["predictions"],
        baseline["metrics"],
        baseline["config"],
    )


def test_comparable_artifacts_produce_deterministic_paired_bootstrap(tmp_path: Path) -> None:
    candidate = _write_side(tmp_path, "candidate", REFERENCES, model_name="Qwen/Qwen3-0.6B")
    baseline = _write_side(
        tmp_path,
        "baseline",
        ("sai hoàn toàn", "không liên quan", "nội dung khác", "dự đoán sai"),
        model_name="some/generic-baseline",
    )

    first = _compare(candidate, baseline)
    second = _compare(candidate, baseline)

    assert first["comparable"] is True
    assert first["paired_bootstrap"]["samples"] == 10_000
    assert first["paired_bootstrap"]["seed"] == 1729
    assert first["paired_bootstrap"]["rouge"] == second["paired_bootstrap"]["rouge"]
    assert first["paired_bootstrap"]["rouge"]["rouge2"]["ci95_low"] > 0.0
    assert len(first["paired_per_example_rouge"]) == len(REFERENCES)
    assert first["paired_per_example_rouge"][0]["id"] == "test_000"
    assert first["paired_per_example_rouge"][0]["delta"]["rouge2"] > 0.0
    assert first["protocol"]["rouge_distribution_version"] == "1.0.0"
    assert first["candidate_superior_all_rouge_with_95pct_paired_support"] is True
    assert first["baseline"]["actual_prediction_artifact_bound_to_metrics"] is True
    # A generic baseline can never be silently relabelled as T5Gemma.
    assert first["t5gemma_claim"] == {
        "eligible": False,
        "point_estimate_beaten": False,
        "beaten": False,
        "reason_if_ineligible": (
            "Requires comparable on-disk predictions, bound metrics/config, and T5Gemma identity in both config and metrics"
        ),
    }


def test_t5gemma_claim_requires_and_uses_actual_identified_artifacts(tmp_path: Path) -> None:
    candidate = _write_side(tmp_path, "candidate", REFERENCES, model_name="Qwen/Qwen3-0.6B")
    baseline = _write_side(
        tmp_path,
        "baseline",
        ("sai hoàn toàn", "không liên quan", "nội dung khác", "dự đoán sai"),
        model_name="google/t5gemma-2-1b-1b",
    )

    result = _compare(candidate, baseline)

    assert result["comparable"] is True
    assert result["baseline"]["identified_as_t5gemma_by_config_and_metrics"] is True
    assert result["t5gemma_claim"]["eligible"] is True
    assert result["t5gemma_claim"]["point_estimate_beaten"] is True
    assert result["t5gemma_claim"]["beaten"] is True


@pytest.mark.parametrize(
    "mutation,expected_gate",
    [
        ("id", "aligned_ids_and_order"),
        ("reference", "aligned_references"),
        ("fingerprint", "same_dataset_or_subset_fingerprint"),
        ("generation", "generation_matches_each_resolved_config"),
        ("backend", "same_supported_evaluation_backend"),
        ("metrics_score", "metrics_recomputed_from_prediction_artifacts"),
    ],
)
def test_comparability_mismatches_fail_closed(tmp_path: Path, mutation: str, expected_gate: str) -> None:
    candidate = _write_side(tmp_path, "candidate", REFERENCES, model_name="Qwen/Qwen3-0.6B")
    baseline = _write_side(tmp_path, "baseline", REFERENCES, model_name="google/t5gemma-2-1b-1b")

    if mutation in {"id", "reference"}:
        rows = [json.loads(line) for line in baseline["predictions"].read_text(encoding="utf-8").splitlines()]
        if mutation == "id":
            rows[0]["id"] = "different-id"
        else:
            rows[0]["reference"] = "một reference khác"
        baseline["predictions"].write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    elif mutation == "fingerprint":
        metrics = json.loads(baseline["metrics"].read_text(encoding="utf-8"))
        metrics["test_data_fingerprint"] = {"num_examples": 4, "sha256": "b" * 64}
        baseline["metrics"].write_text(json.dumps(metrics), encoding="utf-8")
    elif mutation == "generation":
        metrics = json.loads(baseline["metrics"].read_text(encoding="utf-8"))
        metrics["generation"]["repetition_penalty"] = 1.15
        baseline["metrics"].write_text(json.dumps(metrics), encoding="utf-8")
    elif mutation == "backend":
        metrics = json.loads(baseline["metrics"].read_text(encoding="utf-8"))
        metrics["rouge_backend"] = "rouge-score==0.1.2"
        baseline["metrics"].write_text(json.dumps(metrics), encoding="utf-8")
    elif mutation == "metrics_score":
        metrics = json.loads(baseline["metrics"].read_text(encoding="utf-8"))
        metrics["rouge2"] += 5.0
        baseline["metrics"].write_text(json.dumps(metrics), encoding="utf-8")

    result = _compare(candidate, baseline)

    assert result["comparable"] is False
    assert result["comparability_gates"][expected_gate] is False
    assert result["paired_bootstrap"]["rouge"] is None
    assert result["paired_per_example_rouge"] is None
    assert result["candidate_strictly_higher_all_rouge_point_estimates"] is False
    assert result["t5gemma_claim"]["eligible"] is False
    assert result["t5gemma_claim"]["beaten"] is False


def test_fingerprints_may_be_absent_only_from_both_metrics(tmp_path: Path) -> None:
    candidate = _write_side(tmp_path, "candidate", REFERENCES, model_name="Qwen/Qwen3-0.6B")
    baseline = _write_side(tmp_path, "baseline", REFERENCES, model_name="some/baseline")
    for side in (candidate, baseline):
        metrics = json.loads(side["metrics"].read_text(encoding="utf-8"))
        metrics.pop("test_data_fingerprint")
        side["metrics"].write_text(json.dumps(metrics), encoding="utf-8")

    result = _compare(candidate, baseline)

    assert result["comparable"] is True
    assert result["fingerprint_status"].startswith("absent_in_both_metrics")
    assert result["comparability_gates"]["aligned_ids_and_order"] is True
    assert result["comparability_gates"]["aligned_references"] is True


def test_missing_baseline_prediction_file_cannot_fall_back_to_hardcoded_target(tmp_path: Path) -> None:
    candidate = _write_side(tmp_path, "candidate", REFERENCES, model_name="Qwen/Qwen3-0.6B")
    baseline = _write_side(tmp_path, "baseline", REFERENCES, model_name="google/t5gemma-2-1b-1b")
    baseline["predictions"].unlink()

    with pytest.raises(FileNotFoundError):
        _compare(candidate, baseline)


def test_reports_length_and_repetition_diagnostics(tmp_path: Path) -> None:
    candidate = _write_side(
        tmp_path,
        "candidate",
        tuple(reference + " lặp lặp lặp lặp" for reference in REFERENCES),
        model_name="Qwen/Qwen3-0.6B",
    )
    baseline = _write_side(tmp_path, "baseline", REFERENCES, model_name="some/baseline")

    result = _compare(candidate, baseline)

    assert result["comparable"] is True
    candidate_diagnostics = result["candidate"]["diagnostics"]
    assert candidate_diagnostics["length_ratio_mean"] > 1.0
    assert candidate_diagnostics["repeated_trigram_rate_mean"] > 0.0
    assert result["candidate_minus_baseline_diagnostics"]["length_ratio_mean"] > 0.0
    assert result["candidate_minus_baseline_diagnostics"]["repeated_trigram_rate_mean"] > 0.0
