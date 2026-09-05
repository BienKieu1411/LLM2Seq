import json

import pytest
from eviseq_afmr.data.prepare_dataset import prepare_dataset


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_pubmed_style_three_split_prepare_writes_report_and_canonical_rows(tmp_path):
    source = tmp_path / "pubmed"
    output = tmp_path / "processed"
    source.mkdir()
    _write(source / "train.label.jsonl", [{"id": "tr", "text": ["a.", "b."], "summary": ["a."]}])
    _write(source / "val.label.jsonl", [{"article_id": "va", "text": "c.", "summary": "c."}])
    _write(source / "test.label.jsonl", [{"id": "te", "text": "d.", "summary": "d."}])
    report = prepare_dataset(source, output, dataset="pubmed")
    assert report["splits"]["validation"]["kept"] == 1
    row = json.loads((output / "validation.jsonl").read_text(encoding="utf-8"))
    assert row["id"] == "va"
    assert set(row) == {"id", "text", "summary", "task", "dataset"}
    assert (output / "preparation_report.json").is_file()


def test_prepare_rejects_cross_split_source_content(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        _write(source / name, [{"id": name, "source": "same", "target": "summary"}])
    with pytest.raises(ValueError, match="Cross-split content leakage"):
        prepare_dataset(source, tmp_path / "processed", dataset="cnndm")
