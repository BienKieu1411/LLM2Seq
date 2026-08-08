from __future__ import annotations

import json
from pathlib import Path

from eviseq.data.arxiv import prepare


def test_prepare_arxiv_reuses_text_summary_schema(tmp_path: Path) -> None:
    source = tmp_path / "arxiv"
    source.mkdir()
    for filename, identifier in (
        ("train.label.jsonl", "arxiv-train"),
        ("val.label.jsonl", "arxiv-validation"),
        ("test.label.jsonl", "arxiv-test"),
    ):
        row = {
            "id": identifier,
            "text": [f"First scientific {identifier} sentence .", "Second sentence ."],
            "summary": ["First abstract sentence .", "Conclusion ."],
        }
        (source / filename).write_text(json.dumps(row) + "\n", encoding="utf-8")

    processed = tmp_path / "processed"
    report = prepare(source, tmp_path / "raw", processed)
    converted = json.loads((processed / "train.jsonl").read_text(encoding="utf-8"))

    assert report["dataset"] == "arxiv"
    assert converted["dataset"] == "arxiv"
    assert converted["source"] == "First scientific arxiv-train sentence.\nSecond sentence."
    assert converted["target"] == "First abstract sentence.\nConclusion."
