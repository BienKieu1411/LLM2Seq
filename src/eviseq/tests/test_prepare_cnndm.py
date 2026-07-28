from __future__ import annotations

import json
from pathlib import Path

import pytest
from eviseq.prepare_cnndm import convert_split, iter_records, prepare


def _write_jsonl(path: Path, records: list[object], *, bom: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    path.write_text(("\ufeff" if bom else "") + payload, encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_prepare_copies_converts_and_writes_reproducible_manifest(tmp_path: Path) -> None:
    source_dir = tmp_path / "incoming"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    _write_jsonl(
        source_dir / "train.txt",
        [
            {
                "id": "train-1",
                "article_text": ["The first sentence .", "The second sentence ."],
                "abstract_text": ["@highlight", "First fact .", "@highlight", "Second fact ."],
            },
            {"id": "train-empty", "article_text": ["Missing summary ."]},
        ],
        bom=True,
    )
    _write_jsonl(
        source_dir / "val.jsonl",
        [
            {
                "doc_id": "validation-1",
                "document": "A validation article .",
                "highlights": "A validation summary .",
            }
        ],
    )
    _write_jsonl(
        source_dir / "test.json",
        [
            {
                "article_id": "test-1",
                "source": ["This", "is", "a", "tokenized", "news", "article", "with", "nine", "tokens"],
                "target": "A short test summary .",
            }
        ],
    )

    report = prepare(source_dir, raw_dir, processed_dir)

    assert report["dataset"] == "cnndm"
    assert report["splits"]["train"]["kept"] == 1
    assert report["splits"]["train"]["skipped"] == 1
    assert report["splits"]["validation"]["kept"] == 1
    assert report["splits"]["test"]["kept"] == 1
    assert report["splits"]["test"]["canonical_fingerprint"]["num_examples"] == 1

    assert (raw_dir / "train.txt").read_bytes() == (source_dir / "train.txt").read_bytes()
    assert (raw_dir / "val.txt").read_bytes() == (source_dir / "val.jsonl").read_bytes()
    assert (raw_dir / "test.txt").read_bytes() == (source_dir / "test.json").read_bytes()

    train = _read_jsonl(processed_dir / "train.jsonl")
    validation = _read_jsonl(processed_dir / "validation.jsonl")
    test = _read_jsonl(processed_dir / "test.jsonl")
    assert train == [
        {
            "id": "train-1",
            "source": "The first sentence.\nThe second sentence.",
            "target": "First fact.\nSecond fact.",
            "task": "summarization",
            "dataset": "cnndm",
        }
    ]
    assert validation[0]["id"] == "validation-1"
    assert validation[0]["source"] == "A validation article."
    # A token list must not become one EviSeq source unit per token; that would
    # corrupt sentence boundaries used by the bridge's salience supervision.
    assert test[0]["source"] == "This is a tokenized news article with nine tokens"
    assert "\n" not in test[0]["source"]

    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == report
    assert prepare(source_dir, raw_dir, processed_dir) == report
    assert json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8")) == report
    assert not list(tmp_path.rglob("*.tmp"))


def test_iter_records_accepts_bom_and_single_line_array_but_rejects_non_objects(tmp_path: Path) -> None:
    valid = tmp_path / "valid.jsonl"
    _write_jsonl(valid, [[{"id": "one"}, {"id": "two"}]], bom=True)
    assert [record["id"] for record in iter_records(valid)] == ["one", "two"]

    invalid = tmp_path / "invalid.jsonl"
    _write_jsonl(invalid, [[{"id": "one"}, 2]])
    with pytest.raises(ValueError, match="must contain JSON objects"):
        list(iter_records(invalid))


def test_iter_records_rejects_pretty_printed_json_array_with_actionable_hint(tmp_path: Path) -> None:
    path = tmp_path / "pretty.json"
    path.write_text('[\n  {"source": "article", "target": "summary"}\n]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="one JSON record per line"):
        list(iter_records(path))


def test_convert_split_is_atomic_when_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "train.txt"
    output = tmp_path / "train.jsonl"
    output.write_text("previous complete output\n", encoding="utf-8")
    _write_jsonl(
        source,
        [
            {"id": "duplicate", "article": "First article.", "summary": "First summary."},
            {"id": "duplicate", "article": "Second article.", "summary": "Second summary."},
        ],
    )

    with pytest.raises(ValueError, match="Duplicate id"):
        convert_split(source, output, "train")

    assert output.read_text(encoding="utf-8") == "previous complete output\n"
    assert not output.with_suffix(".jsonl.tmp").exists()


def test_prepare_rejects_cross_split_id_leakage_and_does_not_publish_manifest(tmp_path: Path) -> None:
    source_dir = tmp_path / "incoming"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    for filename, label in (("train.txt", "train"), ("val.txt", "validation"), ("test.txt", "test")):
        identifier = "leaked-id" if label in {"train", "test"} else "validation-id"
        _write_jsonl(
            source_dir / filename,
            [{"id": identifier, "article": f"{label} article.", "summary": f"{label} summary."}],
        )

    with pytest.raises(ValueError, match="Cross-split ID leakage between train and test"):
        prepare(source_dir, raw_dir, processed_dir)

    assert not (processed_dir / "manifest.json").exists()
    assert not list(tmp_path.rglob("*.tmp"))
