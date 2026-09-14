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


def test_booksum_schema_and_whole_json_arrays_are_canonicalized(tmp_path):
    source = tmp_path / "booksum"
    output = tmp_path / "processed"
    source.mkdir()
    (source / "train.json").write_text(
        json.dumps([{"chapter_id": "ch-1", "chapter": "A long chapter.", "summary_text": "A short summary."}]),
        encoding="utf-8",
    )
    _write(
        source / "val.jsonl",
        [{"chapter_id": "ch-2", "chapter": ["Part one.", "Part two."], "summary_text": "Summary."}],
    )
    _write(source / "test.jsonl", [{"chapter_id": "ch-3", "text": "Chapter.", "summary_text": "Summary."}])

    report = prepare_dataset(source, output, dataset="booksum")

    assert report["splits"]["train"]["kept"] == 1
    row = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    assert row["id"] == "ch-1"
    assert row["text"] == "A long chapter."
    assert row["summary"] == "A short summary."


def test_govreport_merges_gao_and_crs_structured_splits(tmp_path):
    source = tmp_path / "govreport"
    output = tmp_path / "processed"
    source.mkdir()

    def gov_row(identifier):
        return {
            "id": identifier,
            "report": [
                {"section_title": "Findings", "paragraphs": [f"Finding for {identifier}.", "A second finding."]}
            ],
            "highlight": [{"section_title": "Summary", "paragraphs": ["The report summary."]}],
        }

    for split in ("train", "valid", "test"):
        _write(source / f"gao_{split}.jsonl", [gov_row(f"{split}-gao")])
        _write(source / f"crs_{split}.jsonl", [gov_row(f"{split}-crs")])

    report = prepare_dataset(source, output, dataset="govreport")

    assert report["splits"]["validation"]["kept"] == 2
    rows = [json.loads(line) for line in (output / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["id"] for row in rows} == {"GAO_valid-gao", "CRS_valid-crs"}
    assert rows[0]["text"] == "Findings\nFinding for valid-gao.\nA second finding."
    assert rows[0]["summary"] == "Summary\nThe report summary."
