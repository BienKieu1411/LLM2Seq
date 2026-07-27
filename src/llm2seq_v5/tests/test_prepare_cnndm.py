"""Offline tests for the self-contained CNN/DailyMail V5 profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from llm2seq_v5.config import load_config
from llm2seq_v5.prepare_cnndm import find_split, iter_records, prepare


def _write_jsonl(path: Path, rows: list[object], *, bom: bool = False) -> None:
    encoding = "utf-8-sig" if bom else "utf-8"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding=encoding)


def _minimal_source(root: Path) -> Path:
    root.mkdir()
    _write_jsonl(
        root / "train.txt",
        [
            {
                "id": "train-a",
                "article_text": ["First article sentence .", "Second sentence ."],
                "abstract_text": ["@highlight", "First summary sentence ."],
            },
            {
                "id": "train-b",
                "article_text": ["A", "tokenized", "article", "with", "more", "than", "eight", "source", "tokens", "."],
                "abstract_text": ["A", "short", "tokenized", "summary", "with", "enough", "array", "items", "."],
            },
        ],
        bom=True,
    )
    _write_jsonl(
        root / "val.txt",
        [{"doc_id": "val-a", "article": "Validation article .", "highlights": "Validation summary ."}],
    )
    # One-line arrays are accepted without loading the complete train file.
    _write_jsonl(
        root / "test.txt",
        [[{"article_id": "test-a", "document": "Test article .", "summary": "Test summary ."}]],
    )
    return root


def test_prepare_cnndm_copies_converts_and_fingerprints(tmp_path: Path):
    source = _minimal_source(tmp_path / "source")
    raw = tmp_path / "owned_raw"
    processed = tmp_path / "processed"
    report = prepare(source, raw, processed)

    assert report["splits"]["train"]["kept"] == 2
    assert report["splits"]["validation"]["kept"] == 1
    assert report["splits"]["test"]["kept"] == 1
    assert (raw / "train.txt").read_bytes() == (source / "train.txt").read_bytes()

    rows = [json.loads(line) for line in (processed / "train.jsonl").read_text().splitlines()]
    assert rows[0] == {
        "id": "train-a",
        "source": "First article sentence.\nSecond sentence.",
        "target": "First summary sentence.",
        "task": "summarization",
        "dataset": "cnndm",
    }
    # A token array becomes one sentence/unit, not one newline-delimited unit
    # per token, so phrase continuation remains usable.
    assert "\n" not in rows[1]["source"]
    assert rows[1]["source"].endswith(".")

    manifest = json.loads((processed / "manifest.json").read_text())
    assert manifest["splits"]["train"]["canonical_fingerprint"]["num_examples"] == 2
    assert len(manifest["splits"]["test"]["canonical_fingerprint"]["sha256"]) == 64


def test_prepare_cnndm_rejects_cross_split_id_leakage(tmp_path: Path):
    source = _minimal_source(tmp_path / "source")
    _write_jsonl(
        source / "val.txt",
        [{"id": "train-a", "article": "Leaked article", "summary": "Leaked summary"}],
    )
    with pytest.raises(ValueError, match="Cross-split ID leakage"):
        prepare(source, tmp_path / "raw", tmp_path / "processed")


def test_cnndm_reader_rejects_pretty_printed_arrays(tmp_path: Path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps([{"article": "a", "summary": "b"}], indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="one JSON record per line"):
        list(iter_records(path))


def test_find_split_uses_documented_precedence(tmp_path: Path):
    (tmp_path / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "train.txt").write_text("{}\n", encoding="utf-8")
    assert find_split(tmp_path, "train").name == "train.txt"
    with pytest.raises(FileNotFoundError, match="Missing test split"):
        find_split(tmp_path, "test")


def test_cnndm_config_is_six_epoch_non_wikilingua_profile():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/cnndm_qwen3_embedding_0_6b_phrase_continuation_4096.yaml")
    assert config["adapter"]["num_bidirectional_layers"] == 6
    assert config["training"]["interface_warmup_epochs"] == 1
    assert config["training"]["full_finetune_epochs"] == 5
    assert config["training"]["interface_warmup_epochs"] + config["training"]["full_finetune_epochs"] == 6
    assert config["training"]["batch_size"] == 32
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["data"]["max_source_length"] == 4096
    assert config["data"]["max_target_length"] == 256
    assert config["data"]["clean_wikihow_metadata"] is False
    assert config["generation"]["min_new_tokens"] == 8
    assert config["generation"]["max_new_tokens"] == 192
    assert "data" not in config["benchmark"]
    assert "diagnostic" not in config["benchmark"]
    assert "paper" not in config["benchmark"]
    assert config["benchmark"]["reference_only"]["rouge2"] == 19.396

    t5_config = yaml.safe_load((root.parent / "T5Gemma/configs/cnndm_full_1b_1b_4096.yaml").read_text(encoding="utf-8"))
    assert config["data"]["source_prefix"] == t5_config["data"]["source_prefix"]
    assert config["data"]["max_source_length"] == t5_config["data"]["max_source_length"]
    assert config["generation"]["max_new_tokens"] == t5_config["generation"]["max_new_tokens"]
    assert config["generation"]["min_new_tokens"] == t5_config["generation"]["min_new_tokens"]


def test_cnndm_modes_are_exposed_without_uploads():
    root = Path(__file__).parents[1]
    script = (root / "run.sh").read_text(encoding="utf-8")
    for mode in ("cnndm-prepare", "cnndm-smoke", "cnndm|cnndm-full", "cnndm-train", "cnndm-eval"):
        assert mode in script
    assert "push_to_hub" not in script
