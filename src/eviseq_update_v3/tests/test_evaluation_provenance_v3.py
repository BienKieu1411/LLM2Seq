"""A resumed ROUGE run must describe the requested checkpoint and source data."""

import copy
import json
import shutil
from pathlib import Path

import pytest
import torch
from eviseq_update_v3.config import resolve_path
from eviseq_update_v3.evaluation.provenance import ensure_evaluation_manifest, evaluation_identity
from eviseq_update_v3.modeling.model import EviSeqAFMR
from eviseq_update_v3.runtime import _write_resolved_config, build_loaders, evaluate
from eviseq_update_v3.training.checkpoint import save_checkpoint
from test_planned_semantic_v3 import config_planned


@pytest.fixture
def run(tmp_path):
    config = config_planned(reviewed=True)
    config["generation"]["max_new_tokens"] = 4
    source = tmp_path / "source.jsonl"
    shutil.copyfile(resolve_path(config["data"]["test_file"], config), source)
    config["data"]["test_file"] = str(source)
    _write_resolved_config(config, tmp_path)
    config_path = tmp_path / "resolved_config.yaml"
    checkpoint = tmp_path / "model.pt"
    model = EviSeqAFMR(config)
    save_checkpoint(checkpoint, model, None, config, epoch=1, step=0)
    predictions = tmp_path / "predictions.jsonl"
    return config, config_path, model, checkpoint, predictions


def must_not_load_model(*args, **kwargs):
    pytest.fail("Model construction must not run when validating existing predictions")


@pytest.mark.parametrize("partial", [False, True])
def test_changed_checkpoint_rejected_before_loading_or_appending(run, monkeypatch, partial):
    config, path, model, checkpoint, predictions = run
    evaluate(path, checkpoint, predictions, device="cpu", max_examples=1 if partial else 0)
    previous = predictions.read_bytes()
    # Keep the filename and architecture; the saved weight bytes change.
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    save_checkpoint(checkpoint, model, None, config, epoch=2, step=1)
    monkeypatch.setattr("eviseq_update_v3.runtime.EviSeqAFMR", must_not_load_model)
    with pytest.raises(ValueError, match="provenance mismatch.*checkpoint_sha256"):
        evaluate(path, checkpoint, predictions, device="cpu")
    assert predictions.read_bytes() == previous


def test_complete_predictions_still_require_existing_checkpoint(run, monkeypatch):
    _, path, _, checkpoint, predictions = run
    evaluate(path, checkpoint, predictions, device="cpu")
    checkpoint.unlink()
    monkeypatch.setattr("eviseq_update_v3.runtime.EviSeqAFMR", must_not_load_model)
    with pytest.raises(FileNotFoundError):
        evaluate(path, checkpoint, predictions, device="cpu")


def test_unverified_legacy_predictions_are_not_accepted(run, monkeypatch):
    _, path, _, checkpoint, predictions = run
    evaluate(path, checkpoint, predictions, device="cpu")
    Path(str(predictions) + ".manifest.json").unlink()
    previous = predictions.read_bytes()
    monkeypatch.setattr("eviseq_update_v3.runtime.EviSeqAFMR", must_not_load_model)
    with pytest.raises(ValueError, match="no evaluation manifest"):
        evaluate(path, checkpoint, predictions, device="cpu")
    assert predictions.read_bytes() == previous


def test_same_checkpoint_resumes_partial_and_complete_with_different_batch_size(run, monkeypatch):
    _, path, _, checkpoint, predictions = run
    evaluate(path, checkpoint, predictions, device="cpu", batch_size=1, max_examples=1)
    prefix = predictions.read_bytes()
    result = evaluate(path, checkpoint, predictions, device="cpu", batch_size=2)
    assert predictions.read_bytes().startswith(prefix)
    assert len(predictions.read_text().splitlines()) == 2
    monkeypatch.setattr("eviseq_update_v3.runtime.EviSeqAFMR", must_not_load_model)
    assert evaluate(path, checkpoint, predictions, device="cpu", batch_size=1) == result


@pytest.mark.parametrize("change", ["source", "generation", "architecture"])
def test_changed_eval_inputs_rejected_even_when_ids_and_references_match(run, monkeypatch, change):
    config, path, _, checkpoint, predictions = run
    evaluate(path, checkpoint, predictions, device="cpu")
    if change == "source":
        source = Path(config["data"]["test_file"])
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        rows[0][config["data"]["source_field"]] += " changed source"
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        if change == "generation":
            config["generation"]["max_new_tokens"] += 1
        else:
            config["decoder"]["grounded_copy"]["semantic_read"]["head_gate_position"] = "pre_norm"
        _write_resolved_config(config, path.parent)
    monkeypatch.setattr("eviseq_update_v3.runtime.EviSeqAFMR", must_not_load_model)
    with pytest.raises(ValueError, match="provenance mismatch"):
        evaluate(path, checkpoint, predictions, device="cpu")


def test_tokenizer_and_precision_are_part_of_identity_but_training_schedule_is_not(run):
    config, _, _, checkpoint, predictions = run
    loader = build_loaders(config, split="test")["test"]

    def identity(cfg=config, device="cpu"):
        return evaluation_identity(cfg, checkpoint, loader.dataset, loader.collate_fn, "test", device)

    original = identity()
    modified = copy.deepcopy(config)
    modified["training"]["full_finetune_epochs"] += 5
    modified["generation"]["batch_size"] += 1
    modified["experiment"]["output_dir"] = "elsewhere"
    assert identity(modified) == original
    assert identity(device="cuda") != original
    loader.collate_fn.decoder_tokenizer.pad_token_id = 99
    assert identity()["decoder_tokenizer_sha256"] != original["decoder_tokenizer_sha256"]
    ensure_evaluation_manifest(predictions, original)
    assert json.loads(Path(str(predictions) + ".manifest.json").read_text()) == original
    # An empty output has no predictions to misattribute; a new identity is safe.
    predictions.touch()
    ensure_evaluation_manifest(predictions, identity())
