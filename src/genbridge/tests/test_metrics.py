import json
import tempfile
from pathlib import Path

import pytest
import torch
from genbridge.compare import compare_predictions
from genbridge.evaluate import (
    _decode_performance,
    _generated_lengths,
    _generation_diagnostics,
    _plan_gate_diagnostics,
    _rouge,
    _salience_diagnostics,
)


def test_standard_rouge_exact_match():
    metrics = _rouge(["Tóm tắt sự kiện chính."], ["Tóm tắt sự kiện chính."])
    assert metrics == {
        "rouge1": 100.0,
        "rouge2": 100.0,
        "rougeL": 100.0,
        "rougeLsum": 100.0,
    }


def test_generation_diagnostics_expose_repetition_and_empty_outputs():
    metrics = _generation_diagnostics(
        ["lặp lặp lặp lặp", ""],
        ["một hai ba bốn", "một hai"],
        ["một hai ba bốn năm sáu", "một hai ba"],
    )
    assert metrics["prediction_words_mean"] == 2.0
    assert metrics["empty_prediction_rate"] == 50.0
    assert metrics["too_short_rate"] == 50.0
    assert metrics["repeated_trigram_rate_mean"] == 25.0
    assert metrics["unique_prediction_rate"] == 100.0
    assert metrics["dominant_prefix_5gram_rate"] == 100.0


def test_generation_diagnostics_detect_common_prefix_collapse():
    predictions = [
        f"hãy đi khám bệnh ngay phần khác {index}"
        for index in range(10)
    ]
    metrics = _generation_diagnostics(
        predictions,
        ["tóm tắt tham chiếu"] * 10,
        ["văn bản nguồn dài"] * 10,
    )
    assert metrics["unique_prediction_rate"] == 100.0
    assert metrics["dominant_prefix_5gram_rate"] == 100.0


def test_salience_diagnostics_report_threshold_and_ranking_quality():
    metrics = _salience_diagnostics(
        [0.9, 0.8, 0.2, 0.1],
        [1, 0, 1, 0],
    )
    assert metrics["salience_precision"] == 50.0
    assert metrics["salience_recall"] == 50.0
    assert metrics["salience_f1"] == 50.0
    assert metrics["salience_average_precision"] == 83.3333
    assert metrics["salience_predicted_positive_rate"] == 50.0
    assert metrics["salience_oracle_positive_rate"] == 50.0
    assert metrics["salience_scored_units"] == 4


def test_decode_metrics_count_through_first_eos_and_aggregate_batch_time():
    generated = torch.tensor([[4, 5, 2, 2], [7, 8, 9, 10]])
    lengths = _generated_lengths(generated, eos_token_id=2)
    assert lengths == [3, 4]
    # Two per-example shares of one 0.4-second batch sum back to 0.4 seconds.
    metrics = _decode_performance([0.2, 0.2], lengths)
    assert metrics["decode_elapsed_seconds"] == 0.4
    assert metrics["generated_tokens_total"] == 7
    assert metrics["decode_examples_per_second"] == 5.0


def test_plan_gate_diagnostics_report_layers_and_generation_regions():
    metrics = _plan_gate_diagnostics(
        layer_sums=[2.0, 6.0],
        layer_counts=[10.0, 10.0],
        step_sums=[1.0] * 16 + [4.0],
        step_counts=[4.0] * 17,
    )
    assert metrics["plan_gate_per_cross_layer"] == [0.2, 0.6]
    assert metrics["plan_gate_early_16_mean"] == 0.25
    assert metrics["plan_gate_late_after_16_mean"] == 1.0


def _write_predictions(path, predictions, references):
    path.write_text(
        "".join(
            json.dumps(
                {"id": f"test_{index}", "prediction": prediction, "reference": reference},
                ensure_ascii=False,
            )
            + "\n"
            for index, (prediction, reference) in enumerate(zip(predictions, references))
        ),
        encoding="utf-8",
    )


def test_paired_comparison_reports_positive_delta_for_better_candidate():
    references = [
        "chọn nguyên liệu rồi nấu món ăn",
        "đọc hướng dẫn và làm đúng thứ tự",
        "sạc thiết bị rồi khởi động lại",
    ]
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / "candidate.jsonl"
        baseline = Path(directory) / "baseline.jsonl"
        _write_predictions(candidate, references, references)
        _write_predictions(
            baseline,
            ["nội dung hoàn toàn khác"] * len(references),
            references,
        )
        result = compare_predictions(
            candidate,
            baseline,
            bootstrap_samples=200,
            randomization_samples=200,
            seed=5,
        )
        assert result["num_examples"] == 3
        assert result["metrics"]["rouge2"]["candidate"] == 100.0
        assert result["metrics"]["rouge2"]["delta"] > 0
        assert result["metrics"]["rouge2"]["delta_ci95"][0] > 0


def test_paired_comparison_rejects_reference_mismatch():
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / "candidate.jsonl"
        baseline = Path(directory) / "baseline.jsonl"
        _write_predictions(candidate, ["a b c"], ["a b c"])
        _write_predictions(baseline, ["a b c"], ["reference khác"])
        with pytest.raises(ValueError, match="Reference mismatch"):
            compare_predictions(
                candidate,
                baseline,
                bootstrap_samples=10,
                randomization_samples=10,
            )
