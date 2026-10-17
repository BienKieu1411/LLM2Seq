import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from fixtures import settings_file
from seam_length_grpo.evaluation import summarize


def test_eval_cli_reloads_standard_checkpoint_and_requires_word_request(tmp_path, monkeypatch):
    settings, tokenizer, path = settings_file(tmp_path, source_weight=0)
    source = Path(__file__).resolve().parents[1] / "evaluate.py"
    spec = importlib.util.spec_from_file_location("length_eval_cli", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_tokenizers", lambda config: (tokenizer, tokenizer))
    output = tmp_path / "predictions.jsonl"
    argv = [
        str(source),
        "--config",
        str(path),
        "--checkpoint",
        settings["sft_checkpoint"],
        "--output",
        str(output),
        "--device",
        "cpu",
        "--batch-size",
        "2",
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(ValueError, match="target_words"):
        module.main()
    monkeypatch.setattr("sys.argv", [*argv, "--target-words", "10"])
    module.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["target_words"] == 10 and row["word_request_origin"] == "external_cli" for row in rows)
    metrics = json.loads(output.with_suffix(".metrics.json").read_text())
    assert metrics["count"] == 2
    with pytest.raises(FileExistsError):
        module.main()
    orphan = tmp_path / "orphan.jsonl"
    orphan.with_suffix(".metrics.json").write_text("keep existing metrics")
    orphan_argv = [*argv]
    orphan_argv[orphan_argv.index("--output") + 1] = str(orphan)
    monkeypatch.setattr("sys.argv", [*orphan_argv, "--target-words", "10"])
    with pytest.raises(FileExistsError):
        module.main()
    assert not orphan.exists()
    assert orphan.with_suffix(".metrics.json").read_text() == "keep existing metrics"
    # An external request supersedes a malformed/missing target_words field;
    # an explicit oracle is a separately labelled protocol.
    test_file = tmp_path / "test.jsonl"
    test_file.write_text(
        "".join(
            json.dumps({**json.loads(line), "target_words": 77}) + "\n" for line in test_file.read_text().splitlines()
        )
    )
    monkeypatch.setattr("sys.argv", [*argv, "--oracle-reference-length", "--overwrite"])
    module.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["target_words"] for row in rows] == [2, 12]
    assert all(row["word_request_origin"] == "reference_length" for row in rows)
    # Evaluation does not require the historical SFT checkpoint still to exist.
    raw = yaml.safe_load(path.read_text())
    raw["sft_checkpoint"] = str(tmp_path / "missing_sft.pt")
    path.write_text(yaml.safe_dump(raw))
    module.main()


def test_checkpoint_selection_requires_every_dataset_constraint():
    def row(domain, quality, cost):
        return {
            "dataset": domain,
            "quality": quality,
            "reward": quality,
            "rouge1": quality,
            "rouge2": quality,
            "rougeL": quality,
            "abs_len_loss": cost * 10,
            "length_cost": cost,
            "length_deviation": 0,
        }

    metrics = summarize([row("large", 0.9, 0.1)] * 20 + [row("small", 0.1, 1.2)])
    assert metrics["macro_quality"] == pytest.approx(0.5)
    assert not metrics["all_length_constraints_met"]
    assert metrics["max_length_violation"] == pytest.approx(0.2)
