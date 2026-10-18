import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "mix_summary_wikilingua.py"


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_wikilingua_train_holdout_is_deterministic_and_source_disjoint(tmp_path):
    wiki_dir = tmp_path / "wikilingua"
    wiki_dir.mkdir()
    train_rows = [
        {"id": f"wiki-{index}", "text": f"source {index}", "summary": f"summary {index}"} for index in range(20)
    ]
    train_rows.append({"id": "wiki-duplicate", "text": "source 0", "summary": "duplicate summary"})
    _write_jsonl(wiki_dir / "train.jsonl", train_rows)
    test_rows = [{"id": "test", "text": "test source", "summary": "test summary"}]
    _write_jsonl(wiki_dir / "test.jsonl", test_rows)

    extra_train = tmp_path / "extra.jsonl"
    _write_jsonl(
        extra_train,
        [
            {"id": "extra", "text": "extra source", "summary": "extra summary"},
            {"id": "test-copy", "text": "test source", "summary": "must be excluded"},
            {"id": "wiki-copy", "text": "source 0", "summary": "duplicate source"},
        ],
    )

    outputs = [tmp_path / "mixed-a", tmp_path / "mixed-b"]
    for output_dir in outputs:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--other-train",
                str(extra_train),
                "--wikilingua-dir",
                str(wiki_dir),
                "--output-dir",
                str(output_dir),
                "--wikilingua-validation-fraction",
                "0.25",
                "--seed",
                "19",
            ],
            check=True,
            capture_output=True,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    part for part in (str(SCRIPT.parents[1]), os.environ.get("PYTHONPATH", "")) if part
                ),
            },
            text=True,
        )

    train = _read_jsonl(outputs[0] / "train.jsonl")
    validation = _read_jsonl(outputs[0] / "validation.jsonl")
    test = _read_jsonl(outputs[0] / "test.jsonl")
    report = json.loads((outputs[0] / "mix_report.json").read_text(encoding="utf-8"))

    assert report["validation_source"] == "deterministic source-group split from WikiLingua train"
    assert report["validation_records"] == len(validation)
    assert 0.20 <= report["validation_fraction_realized"] <= 0.35
    assert {row["text"] for row in train}.isdisjoint({row["text"] for row in validation})
    assert {row["text"] for row in train}.isdisjoint({row["text"] for row in test})
    assert {row["text"] for row in validation}.isdisjoint({row["text"] for row in test})
    assert any(row["text"] == "extra source" for row in train)
    assert not any(row["text"] == "test source" for row in train)
    assert (outputs[0] / "validation.jsonl").read_bytes() == (outputs[1] / "validation.jsonl").read_bytes()
    assert (outputs[0] / "train.jsonl").read_bytes() == (outputs[1] / "train.jsonl").read_bytes()
    assert (outputs[0] / "test.jsonl").read_bytes() == (outputs[1] / "test.jsonl").read_bytes()


def test_external_validation_json_is_canonicalized_and_excluded_from_train(tmp_path):
    wiki_dir = tmp_path / "wikilingua"
    wiki_dir.mkdir()
    _write_jsonl(
        wiki_dir / "train.jsonl",
        [
            {"id": f"wiki-{index}", "text": f"wiki source {index}", "summary": f"wiki summary {index}"}
            for index in range(4)
        ],
    )
    _write_jsonl(wiki_dir / "test.jsonl", [{"id": "test", "text": "test source", "summary": "test summary"}])
    validation_file = tmp_path / "val_clean.json"
    validation_file.write_text(
        json.dumps(
            [
                {"id": "val-1", "input": "validation source", "output": "validation summary"},
                {"id": "val-2", "input": "test source", "output": "overlaps test and must fail"},
            ]
        ),
        encoding="utf-8",
    )
    train_file = tmp_path / "train_clean.jsonl"
    _write_jsonl(
        train_file,
        [
            {"id": "train-1", "text": "additional source", "summary": "additional summary"},
            {"id": "train-val-copy", "text": "validation source", "summary": "must be excluded"},
            {"id": "train-test-copy", "text": "test source", "summary": "must be excluded"},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prepared-train",
            str(train_file),
            "--wikilingua-dir",
            str(wiki_dir),
            "--output-dir",
            str(tmp_path / "mixed"),
            "--validation-file",
            str(validation_file),
            "--validation-source-field",
            "input",
            "--validation-target-field",
            "output",
        ],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(SCRIPT.parents[1]), os.environ.get("PYTHONPATH", "")))},
        text=True,
    )
    assert result.returncode != 0
    assert "shares 1 exact source document" in result.stderr

    validation_file.write_text(
        json.dumps([{"id": "val-1", "input": "validation source", "output": "validation summary"}]),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prepared-train",
            str(train_file),
            "--wikilingua-dir",
            str(wiki_dir),
            "--output-dir",
            str(tmp_path / "mixed"),
            "--validation-file",
            str(validation_file),
            "--validation-source-field",
            "input",
            "--validation-target-field",
            "output",
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(SCRIPT.parents[1]), os.environ.get("PYTHONPATH", "")))},
        text=True,
    )
    mixed = tmp_path / "mixed"
    report = json.loads((mixed / "mix_report.json").read_text(encoding="utf-8"))
    assert report["validation_split"] == "external validation file"
    assert _read_jsonl(mixed / "validation.jsonl") == [
        {"id": "external_validation:1:val-1", "text": "validation source", "summary": "validation summary"}
    ]
    assert not any(row["text"] == "validation source" for row in _read_jsonl(mixed / "train.jsonl"))
    assert not any(row["text"] == "test source" for row in _read_jsonl(mixed / "train.jsonl"))


def test_sds_mds_replace_batch_files_and_only_icons_are_removed(tmp_path):
    wiki_dir = tmp_path / "wikilingua"
    wiki_dir.mkdir()
    _write_jsonl(
        wiki_dir / "train.jsonl",
        [
            {"id": f"wiki-{index}", "text": f"wiki source {index}", "summary": f"wiki summary {index}"}
            for index in range(5)
        ],
    )
    _write_jsonl(wiki_dir / "validation.jsonl", [{"id": "val", "text": "val source", "summary": "val summary"}])
    _write_jsonl(wiki_dir / "test.jsonl", [{"id": "test", "text": "test source", "summary": "test summary"}])

    data_dir = tmp_path / "summary_data"
    data_dir.mkdir()
    _write_jsonl(
        data_dir / "sds.jsonl",
        [
            {
                "id": "sds-1",
                "input": "Read 😊 more at https://example.org",
                "abstract_sum": "A ✅ short summary",
            },
            {"id": "sds-2", "input": "Another 🧪 source", "abstract_sum": "Another summary"},
        ],
    )
    _write_jsonl(
        data_dir / "mds.jsonl",
        [{"id": "mds-1", "input": "Multi-document 📚 source", "abstract_sum": "Combined ★ result"}],
    )
    # The old batch-file convention is deliberately ignored.
    _write_jsonl(data_dir / "batch1.jsonl", [{"input": "legacy source", "output": "legacy summary"}])

    mixed = tmp_path / "mixed"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary-data-dir",
            str(data_dir),
            "--wikilingua-dir",
            str(wiki_dir),
            "--output-dir",
            str(mixed),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(SCRIPT.parents[1]), os.environ.get("PYTHONPATH", "")))},
        text=True,
    )

    train = _read_jsonl(mixed / "train.jsonl")
    sds = next(row for row in train if row["id"].startswith("sds:"))
    mds = next(row for row in train if row["id"].startswith("mds:"))
    assert sds["text"] == "Read more at https://example.org"
    assert sds["summary"] == "A short summary"
    assert mds["text"] == "Multi-document source"
    assert mds["summary"] == "Combined result"
    assert not any("legacy" in row["text"] for row in train)

    report = json.loads((mixed / "mix_report.json").read_text(encoding="utf-8"))
    by_label = {row["label"]: row for row in report["train_sources"]}
    assert by_label["sds"]["icons_removed"] == {"source": 2, "target": 1, "records_with_icons": 2}
    assert by_label["sds"]["records_kept"] == 2
    assert by_label["sds"]["records_skipped_empty_source_or_target"] == 0
    assert by_label["mds"]["icons_removed"] == {"source": 1, "target": 1, "records_with_icons": 1}


def test_caps_on_validation_summary_wikilingua_vietnews_and_unlimited_others(tmp_path):
    wiki_dir = tmp_path / "wikilingua"
    wiki_dir.mkdir()
    _write_jsonl(
        wiki_dir / "train.jsonl",
        [{"id": f"wiki-{i}", "text": f"wiki source {i}", "summary": f"wiki summary {i}"} for i in range(12)],
    )
    _write_jsonl(wiki_dir / "test.jsonl", [{"id": "test", "text": "test source", "summary": "test summary"}])

    val_file = tmp_path / "val.jsonl"
    _write_jsonl(
        val_file,
        [{"id": f"val-{i}", "text": f"val source {i}", "summary": f"val summary {i}"} for i in range(10)],
    )

    data_dir = tmp_path / "summary_data"
    data_dir.mkdir()
    _write_jsonl(
        data_dir / "sds.jsonl",
        [{"id": f"sds-{i}", "input": f"sds source {i}", "abstract_sum": f"sds summary {i}"} for i in range(8)],
    )
    _write_jsonl(
        data_dir / "mds.jsonl",
        [{"id": f"mds-{i}", "input": f"mds source {i}", "abstract_sum": f"mds summary {i}"} for i in range(8)],
    )

    vietnews_file = tmp_path / "vietnews_train.jsonl"
    _write_jsonl(
        vietnews_file,
        [{"id": f"vn-{i}", "text": f"vietnews source {i}", "summary": f"vietnews summary {i}"} for i in range(10)],
    )

    vims_file = tmp_path / "vims_train.jsonl"
    _write_jsonl(
        vims_file,
        [{"id": f"vims-{i}", "text": f"vims source {i}", "summary": f"vims summary {i}"} for i in range(3)],
    )

    vlsp_file = tmp_path / "vlsp_train.jsonl"
    _write_jsonl(
        vlsp_file,
        [{"id": f"vlsp-{i}", "text": f"vlsp source {i}", "summary": f"vlsp summary {i}"} for i in range(4)],
    )

    mixed = tmp_path / "mixed_capped"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary-data-dir",
            str(data_dir),
            "--validation-file",
            str(val_file),
            "--wikilingua-dir",
            str(wiki_dir),
            "--other-train",
            str(vietnews_file),
            "--other-train",
            str(vims_file),
            "--other-train",
            str(vlsp_file),
            "--max-validation-records",
            "4",
            "--max-summary-records",
            "7",
            "--max-wikilingua-records",
            "5",
            "--max-vietnews-records",
            "3",
            "--output-dir",
            str(mixed),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(SCRIPT.parents[1]), os.environ.get("PYTHONPATH", "")))},
        text=True,
    )

    train = _read_jsonl(mixed / "train.jsonl")
    val = _read_jsonl(mixed / "validation.jsonl")
    report = json.loads((mixed / "mix_report.json").read_text(encoding="utf-8"))

    assert len(val) == 4
    assert report["validation_records"] == 4

    by_label = {row["label"]: row for row in report["train_sources"]}
    assert by_label["sds"]["records_kept"] + by_label["mds"]["records_kept"] == 7
    assert by_label["wikilingua"]["records_kept"] == 5
    assert by_label["other_train_1"]["records_kept"] == 3  # vietnews capped at 3
    assert by_label["other_train_2"]["records_kept"] == 3  # vims kept all 3
    assert by_label["other_train_3"]["records_kept"] == 4  # vlsp kept all 4
    assert len(train) == 7 + 5 + 3 + 3 + 4
    assert report["train_total"] == 22

    # Also test inline path:cap syntax for other-train
    mixed2 = tmp_path / "mixed_capped2"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary-data-dir",
            str(data_dir),
            "--validation-file",
            str(val_file),
            "--wikilingua-dir",
            str(wiki_dir),
            "--other-train",
            f"{vietnews_file}:2",
            "--other-train",
            str(vims_file),
            "--other-train",
            str(vlsp_file),
            "--max-validation-records",
            "3",
            "--max-summary-records",
            "5",
            "--max-wikilingua-records",
            "4",
            "--output-dir",
            str(mixed2),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(SCRIPT.parents[1]), os.environ.get("PYTHONPATH", "")))},
        text=True,
    )
    val2 = _read_jsonl(mixed2 / "validation.jsonl")
    report2 = json.loads((mixed2 / "mix_report.json").read_text(encoding="utf-8"))
    assert len(val2) == 3
    by_label2 = {row["label"]: row for row in report2["train_sources"]}
    assert by_label2["other_train_1"]["records_kept"] == 2  # capped via :2
    assert by_label2["other_train_2"]["records_kept"] == 3  # all
    assert by_label2["other_train_3"]["records_kept"] == 4  # all

