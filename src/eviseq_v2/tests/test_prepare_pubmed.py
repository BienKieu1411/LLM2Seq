from __future__ import annotations

import json
from pathlib import Path

import pytest
from eviseq_v2.prepare_pubmed import prepare


def _write(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _source_row(identifier: str) -> dict:
    return {
        "id": identifier,
        "text": ["First biomedical sentence .", "Second sentence ."],
        "summary": ["First abstract sentence .", "Conclusion ."],
        "label": [0, 1],
    }


def test_prepare_pubmed_is_self_contained_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "pubmet"
    for filename, identifier in (
        ("train.label.jsonl", "train-1"),
        ("val.label.jsonl", "val-1"),
        ("test.label.jsonl", "test-1"),
    ):
        _write(source / filename, _source_row(identifier))

    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    report = prepare(source, raw, processed)
    repeated = prepare(source, raw, processed)
    row = json.loads((processed / "train.jsonl").read_text(encoding="utf-8"))

    assert report == repeated
    assert report["dataset"] == "pubmed"
    assert row == {
        "id": "train-1",
        "source": "First biomedical sentence.\nSecond sentence.",
        "target": "First abstract sentence.\nConclusion.",
        "task": "summarization",
        "dataset": "pubmed",
    }
    assert (processed / "manifest.json").is_file()
    assert not list(tmp_path.rglob("*.tmp"))


def test_prepare_pubmed_rejects_cross_split_id_leakage(tmp_path: Path) -> None:
    source = tmp_path / "pubmed"
    _write(source / "train.label.jsonl", _source_row("leaked"))
    _write(source / "val.label.jsonl", _source_row("validation"))
    _write(source / "test.label.jsonl", _source_row("leaked"))

    processed = tmp_path / "processed"
    with pytest.raises(ValueError, match="Cross-split ID leakage between train and test"):
        prepare(source, tmp_path / "raw", processed)

    assert not (processed / "manifest.json").exists()
    assert not list(tmp_path.rglob("*.tmp"))
