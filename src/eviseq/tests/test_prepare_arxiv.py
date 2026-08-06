from __future__ import annotations

import json
from pathlib import Path

from eviseq.data.arxiv import prepare


def test_prepare_arxiv_reuses_text_summary_schema(tmp_path: Path) -> None:
    source = tmp_path / "arxiv"
    source.mkdir()
    row = {
        "id": "arxiv-1",
        "text": ["First scientific sentence .", "Second sentence ."],
        "summary": ["First abstract sentence .", "Conclusion ."],
    }
    for filename in ("train.label.jsonl", "val.label.jsonl", "test.label.jsonl"):
        (source / filename).write_text(json.dumps(row) + "\n", encoding="utf-8")

    processed = tmp_path / "processed"
    report = prepare(source, tmp_path / "raw", processed)
    converted = json.loads((processed / "train.jsonl").read_text(encoding="utf-8"))

    assert report["dataset"] == "arxiv"
    assert converted["dataset"] == "arxiv"
    assert converted["source"] == "First scientific sentence.\nSecond sentence."
    assert converted["target"] == "First abstract sentence.\nConclusion."
