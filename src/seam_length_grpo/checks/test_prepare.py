import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from eviseq_afmr.data.normalization import detokenize
from fixtures import settings_file
from seam_length_grpo.data import LengthDataset
from seam_length_grpo.prepare import annotate
from seam_length_grpo.reward import count_words


def test_prepare_keeps_every_record_and_field_with_own_word_count(tmp_path):
    source, output = tmp_path / "train.jsonl", tmp_path / "train_length.jsonl"
    rows = [
        {"id": "a", "text": "source A 🙂", "summary": "Một bản tóm tắt ngắn.", "metadata": {"keep": True}},
        {"id": "b", "text": "source B", "summary": ["Một.", "Hai ba."], "other": [1, 2]},
    ]
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n\n")
    original = source.read_bytes()
    report = annotate(source, output)
    result = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["target_words"] for row in result] == [5, 3]
    assert report["samples"] == 2 and report["dropped_samples"] == 0
    assert source.read_bytes() == original
    for before, after in zip(rows, result, strict=True):
        assert {key: value for key, value in after.items() if key != "target_words"} == before


def test_prepared_lengths_match_training_detokenization(tmp_path):
    settings, _, _ = settings_file(tmp_path, source_weight=0)
    base = yaml.safe_load((tmp_path / "base.yaml").read_text())
    base["data"]["detokenize"] = True
    source, output = tmp_path / "input.jsonl", tmp_path / "counted.jsonl"
    summary = "tiếng Việt does n't -lrb- tốt -rrb- ."
    source.write_text(json.dumps({"text": "source content", "summary": summary}) + "\n")
    annotate(source, output, normalize=True)
    result = json.loads(output.read_text())
    assert result["summary"] == summary
    dataset = LengthDataset(output, base["data"], settings["length_control"])
    assert dataset[0]["target_words"] == count_words(dataset[0]["reference"]) == count_words(detokenize(summary))


def test_prepare_fails_atomically_without_skipping_bad_rows(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text('{"summary":"valid target"}\n{"summary":""}\n')
    output.write_text("existing file stays intact")
    with pytest.raises(ValueError, match=":2:"):
        annotate(source, output, overwrite=True)
    assert output.read_text() == "existing file stays intact"
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        annotate(source, output)
    with pytest.raises(ValueError, match="separate output"):
        annotate(source, source, overwrite=True)


def test_existing_length_requires_explicit_recompute(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text(json.dumps({"summary": "two words", "target_words": 80}) + "\n")
    with pytest.raises(ValueError, match="replace-length"):
        annotate(source, output)
    assert not output.exists()
    annotate(source, output, replace_length=True)
    assert json.loads(output.read_text())["target_words"] == 2


def test_prepare_cli_custom_fields_works_without_model_directories(tmp_path):
    settings, _, path = settings_file(tmp_path, source_weight=0)
    base_path = tmp_path / "base.yaml"
    base = yaml.safe_load(base_path.read_text())
    base["model"]["encoder_name"], base["model"]["decoder_name"] = "/missing/encoder", "/missing/decoder"
    base["data"]["target_field"] = "abstract_sum"
    base["data"]["detokenize"] = True
    base_path.write_text(yaml.safe_dump(base))
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text(json.dumps({"input": "source unchanged 🙂", "abstract_sum": "một hai ba"}) + "\n")
    script = Path(__file__).resolve().parents[1] / "prepare_data.py"
    result = subprocess.run(
        [sys.executable, str(script), "--input", str(source), "--output", str(output), "--config", str(path)],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text())["target_words"] == 3
    assert json.loads(result.stdout)["detokenize_for_count"] is True


def test_nested_target_and_canonical_target_alias(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text(json.dumps({"nested": {"gold": "one two"}}) + "\n")
    annotate(source, output, target_field="nested.gold")
    assert json.loads(output.read_text())["target_words"] == 2
    source.write_text(json.dumps({"target": "one two three"}) + "\n")
    annotate(source, output, overwrite=True)
    assert json.loads(output.read_text())["target_words"] == 3
