import sys

import pytest
from seam_grpo.output import reset_output


@pytest.mark.parametrize("protected_key", ["sft_checkpoint", "base_config", "train_file", "encoder_name"])
def test_overwrite_rejects_outputs_containing_inputs(tmp_path, protected_key):
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("must survive")
    settings = {
        "output_dir": str(output),
        "base_config": str(tmp_path / "sft.yaml"),
        "sft_checkpoint": str(tmp_path / "sft.pt"),
    }
    base = {
        "data": {"train_file": str(tmp_path / "train.jsonl")},
        "model": {"encoder_name": "__tiny__", "decoder_name": "__tiny__"},
    }
    if protected_key in settings:
        settings[protected_key] = str(sentinel)
    elif protected_key == "train_file":
        base["data"][protected_key] = str(sentinel)
    else:
        base["model"][protected_key] = str(output / "encoder")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        reset_output(settings, base, use_cpu=True)
    assert sentinel.read_text() == "must survive"


def test_dry_run_with_overwrite_does_not_delete_artifacts(tmp_path, monkeypatch, capsys):
    from seam_grpo import train as runtime

    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("previous run")
    monkeypatch.setattr(sys, "argv", ["run_grpo.py", "--config", "unused.yaml", "--dry-run", "--overwrite-output-dir"])
    monkeypatch.setattr(runtime, "load", lambda path: {"output_dir": str(tmp_path), "reward": {}})
    monkeypatch.setattr(runtime, "prepare", lambda settings: {"model": {}, "data": {}})
    runtime.main()
    assert sentinel.read_text() == "previous run"
    assert "grpo:" in capsys.readouterr().out
