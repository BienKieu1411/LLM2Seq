from __future__ import annotations

import json
from pathlib import Path

import pytest

from rouge155.evaluate_rouge import (
    _DETAIL_PATTERN,
    _ROUGE_PROTOCOL,
    _json_sha256,
    _parse_per_example,
    _prepare_data,
    _sha256,
)
from rouge155.paired_bootstrap import compare


def test_detail_parser_accepts_task_peer_ids_in_input_order() -> None:
    lines = []
    for task in (1, 2):
        for metric in ("1", "2", "L"):
            lines.append(f"1 ROUGE-{metric} Eval {task}.1 R:0.10000 P:0.20000 F:0.{task}0000")
    rows = _parse_per_example("\n".join(lines), 2)
    assert [row["row_index"] for row in rows] == [0, 1]
    assert [row["eval_task_id"] for row in rows] == [1, 2]
    assert rows[1]["rouge2"]["f1"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "raw",
    [
        "1 ROUGE-1 Eval 1 R:0.1 P:0.1 F:0.1",
        "1 ROUGE-1 Eval 1.2 R:0.1 P:0.1 F:0.1",
        "1 ROUGE-1 Eval 1.1 R:1.1 P:0.1 F:0.1",
    ],
)
def test_detail_parser_fails_closed_on_invalid_contract(raw: str) -> None:
    with pytest.raises(ValueError):
        _parse_per_example(raw, 1)


def test_prepare_data_uses_zero_padded_names(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps({"id": index, "prediction": f"p {index}", "reference": f"r {index}"}) + "\n"
            for index in range(12)
        ),
        encoding="utf-8",
    )
    work = tmp_path / "rouge_data"
    bindings = _prepare_data(predictions, work, "prediction", "reference")
    assert len(bindings) == 12
    assert (work / "system/summary.000000010.txt").is_file()
    assert sorted(path.name for path in (work / "system").iterdir())[10] == "summary.000000010.txt"
    assert (work / "reference/summary.A.000000010.txt").is_file()


def _write_run(root: Path, name: str, values: list[float], reference_suffix: str = "") -> Path:
    predictions = root / f"{name}.jsonl"
    records = [
        {
            "id": f"row-{index}",
            "prediction": f"prediction {name} {index}",
            "reference": f"reference {index}{reference_suffix}",
        }
        for index in range(len(values))
    ]
    predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    raw = root / f"{name}.per_example.raw.txt"
    raw.write_text("official raw output\n", encoding="utf-8")
    contracts = [_json_sha256({"id": row["id"], "reference": row["reference"]}) for row in records]
    rows = []
    for index, (record, value, contract) in enumerate(zip(records, values, contracts, strict=True)):
        metric = {"recall": value, "precision": value, "f1": value}
        rows.append(
            {
                "row_index": index,
                "eval_task_id": index + 1,
                "id": record["id"],
                "row_contract_sha256": contract,
                "rouge1": dict(metric),
                "rouge2": dict(metric),
                "rougeL": dict(metric),
            }
        )
    details = root / f"{name}.per_example.json"
    details.write_text(
        json.dumps(
            {
                "schema_version": "eviseq.perl_rouge155_details.v1",
                "num_examples": len(values),
                "backend": "Perl ROUGE-1.5.5 via pyrouge==0.1.3",
                "scorer_fingerprint_sha256": "f" * 64,
                "headline_protocol": _ROUGE_PROTOCOL,
                "detail_protocol": f"{_ROUGE_PROTOCOL} -d",
                "prediction_field": "prediction",
                "reference_field": "reference",
                "predictions_file": str(predictions),
                "predictions_sha256": _sha256(predictions),
                "id_reference_sha256": _json_sha256(contracts),
                "raw_detail_file": str(raw),
                "raw_detail_sha256": _sha256(raw),
                "per_example_f1_mean": {
                    "rouge1": sum(values) / len(values),
                    "rouge2": sum(values) / len(values),
                    "rougeL": sum(values) / len(values),
                },
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    mean = sum(values) / len(values) * 100.0
    headline = root / f"{name}.rouge155.json"
    headline.write_text(
        json.dumps(
            {
                "rouge1": mean,
                "rouge2": mean,
                "rougeL": mean,
                "num_examples": len(values),
                "backend": "Perl ROUGE-1.5.5 via pyrouge==0.1.3",
                "scorer_fingerprint_sha256": "f" * 64,
                "pyrouge_default_args": _ROUGE_PROTOCOL,
                "predictions_file": str(predictions),
                "predictions_sha256": _sha256(predictions),
                "per_example_scores_file": str(details),
                "per_example_scores_sha256": _sha256(details),
                "raw_detail_sha256": _sha256(raw),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return headline


def test_paired_bootstrap_is_reproducible_and_detects_positive_delta(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "baseline", [0.10, 0.20, 0.30, 0.40])
    candidate = _write_run(tmp_path, "candidate", [0.20, 0.30, 0.40, 0.50])
    first = compare(candidate, baseline, tmp_path / "first.json", samples=500, seed=1729)
    second = compare(candidate, baseline, tmp_path / "second.json", samples=500, seed=1729)
    assert first["scores"] == second["scores"]
    assert first["scores"]["rouge2"]["official_headline_delta"] == pytest.approx(10.0)
    assert first["scores"]["rouge2"]["paired_mean_delta"] == pytest.approx(10.0)
    assert first["decision"]["rouge2_ci95_low_gt_zero"] is True


def test_paired_bootstrap_rejects_reference_mismatch(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "baseline", [0.1, 0.2, 0.3])
    candidate = _write_run(tmp_path, "candidate", [0.2, 0.3, 0.4], reference_suffix=" changed")
    with pytest.raises(ValueError, match="same ordered IDs/references"):
        compare(candidate, baseline, tmp_path / "comparison.json", samples=100)


def test_detail_regex_does_not_accept_integer_only_eval_id() -> None:
    assert _DETAIL_PATTERN.match("1 ROUGE-1 Eval 1 R:0.1 P:0.1 F:0.1") is None
