"""Offline tests for the V5 PubMed preparation and 4096-token profile."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from llm2seq_v5.config import load_config
from llm2seq_v5.prepare_pubmed import prepare


def _write(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_pubmed_preparation_and_profile(tmp_path: Path) -> None:
    source = tmp_path / "pubmet"
    for filename, identifier in (
        ("train.label.jsonl", "train-1"),
        ("val.label.jsonl", "val-1"),
        ("test.label.jsonl", "test-1"),
    ):
        _write(
            source / filename,
            {
                "id": identifier,
                "text": ["First biomedical sentence .", "Second sentence ."],
                "summary": ["First abstract sentence .", "Conclusion ."],
                "label": [0, 1],
            },
        )

    report = prepare(source, tmp_path / "raw", tmp_path / "processed")
    row = json.loads((tmp_path / "processed/train.jsonl").read_text(encoding="utf-8"))
    assert report["dataset"] == "pubmed"
    assert row["source"] == "First biomedical sentence.\nSecond sentence."
    assert row["target"] == "First abstract sentence.\nConclusion."
    assert row["dataset"] == "pubmed"
    assert "label" not in row

    root = Path(__file__).parents[1]
    config = load_config(root / "configs/pubmed_qwen3_embedding_0_6b_phrase_continuation_4096.yaml")
    t5 = yaml.safe_load((root.parent / "T5Gemma/configs/pubmed_full_1b_1b_4096.yaml").read_text())
    assert config["data"]["max_source_length"] == 4096
    assert config["data"]["max_target_length"] == 512
    assert config["data"]["source_prefix"] == t5["data"]["source_prefix"]
    assert config["training"]["interface_warmup_epochs"] == 2
    assert config["training"]["full_finetune_epochs"] == 6
    assert t5["training"]["num_train_epochs"] == 4
