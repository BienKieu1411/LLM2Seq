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
    assert _read_jsonl(mixed / "validation.jsonl") == [
        {"id": "external_validation:1:val-1", "text": "validation source", "summary": "validation summary"}
    ]
    assert not any(row["text"] == "validation source" for row in _read_jsonl(mixed / "train.jsonl"))
    assert not any(row["text"] == "test source" for row in _read_jsonl(mixed / "train.jsonl"))
