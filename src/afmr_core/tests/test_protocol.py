import json
from pathlib import Path

import pytest
import torch

from afmr_core.config import config_fingerprint, load_config, validate_config
from afmr_core.data.prepare_dataset import prepare_dataset
from afmr_core.data.sampling import CanonicalBatchManifest, DistributedBatchSampler, materialize_global_batches
from afmr_core.evaluation.generate import _sample_token
from afmr_core.evaluation.provenance import ensure_evaluation_manifest
from afmr_core.modeling.model import AFMRModel
from afmr_core.training import checkpoint as checkpoint_module
from afmr_core.training.checkpoint import architecture_spec, load_checkpoint, save_checkpoint
from afmr_core.training.engine import GlobalCosineScheduler, global_cosine_factor
from afmr_core.training.optimizer import _component

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_is_strict_and_local():
    config = load_config(ROOT / "configs" / "afmr_smoke.yaml")
    assert config["architecture"]["name"] == "afmr_value_anchor"
    assert config["decoder"]["grounded_copy"]["semantic_read"]["rank"] == 8
    assert config["generation"]["temperature"] == 0.0
    assert config["generation"]["top_k"] == 0
    assert config["generation"]["top_p"] == 1.0
    assert config_fingerprint(config) == config["_meta"]["config_fingerprint"]
    assert architecture_spec(config)["graph_version"].startswith("afmr_core")


def test_legacy_control_config_is_explicit():
    config = load_config(ROOT / "configs" / "afmr_legacy.yaml")
    copy_config = config["decoder"]["grounded_copy"]
    semantic = copy_config["semantic_read"]
    assert copy_config["enabled"] is True
    assert copy_config["readout_mode"] == "legacy_copy_mixture"
    assert semantic["inner_gate"] is True
    assert semantic["output_init"] == "zero"
    assert semantic["cap_mode"] == "legacy"
    spec = architecture_spec(config)
    assert spec["grounded_copy"]["readout_mode"] == "legacy_copy_mixture"
    assert spec["grounded_copy"]["semantic"]["cap_mode"] == "legacy"


def test_config_rejects_invalid_cross_gate_bounds():
    config = load_config(ROOT / "configs" / "afmr_smoke.yaml")
    config["decoder"]["cross_gate_init"] = 1.0
    with pytest.raises(ValueError, match="cross_gate_init"):
        validate_config(config)


def test_pubmed_runner_fails_fast_without_downloading_models():
    script = (ROOT / "scripts" / "run_pubmed_pair.sh").read_text(encoding="utf-8")
    assert "export HF_HUB_OFFLINE=1" in script
    assert "export TRANSFORMERS_OFFLINE=1" in script
    assert "export ALLOW_CROSS_SPLIT_CONTENT=" in script
    assert 'CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/best.pt}"' in script
    assert "PPLX encoder not found" in script
    assert "Decoder model not found" in script


def test_duplicate_content_flag_is_read_from_environment(tmp_path, monkeypatch):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    rows = {
        "train.jsonl": {"id": "train-1", "text": "same source", "summary": "one"},
        "val.jsonl": {"id": "val-1", "text": "same source", "summary": "two"},
        "test.jsonl": {"id": "test-1", "text": "other source", "summary": "three"},
    }
    for filename, row in rows.items():
        (input_dir / filename).write_text(json.dumps(row) + "\n", encoding="utf-8")

    monkeypatch.delenv("ALLOW_CROSS_SPLIT_CONTENT", raising=False)
    with pytest.raises(ValueError, match="Cross-split content leakage"):
        prepare_dataset(input_dir, tmp_path / "strict", dataset="custom")

    monkeypatch.setenv("ALLOW_CROSS_SPLIT_CONTENT", "true")
    report = prepare_dataset(input_dir, tmp_path / "allowed", dataset="custom")
    assert report["splits"]["validation"]["duplicate_sources"] == 1


def test_canonical_manifest_replays_across_ranks():
    lengths = [9, 2, 5, 7, 1, 4, 8]
    manifest = CanonicalBatchManifest.build(
        lengths,
        global_batch_size=4,
        seed=11,
        epoch=2,
        multiplier=2,
        shuffle=True,
        bucket=True,
    )
    rank0 = list(DistributedBatchSampler(lengths, 2, 0, 2, manifest=manifest))
    rank1 = list(DistributedBatchSampler(lengths, 2, 1, 2, manifest=manifest))
    assert len(rank0) == len(rank1) == len(manifest.batches)
    for global_batch, left, right in zip(manifest.batches, rank0, rank1):
        assert sorted(index for index in (*left, *right) if index >= 0) == sorted(global_batch)
    replay = materialize_global_batches(lengths, 4, seed=11, epoch=2, multiplier=2, shuffle=True, bucket=True)
    assert tuple(tuple(batch) for batch in replay) == manifest.batches


def test_manifest_round_trip(tmp_path):
    manifest = CanonicalBatchManifest.build([1, 2, 3], 2, seed=5, bucket=False)
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = CanonicalBatchManifest.load(path)
    assert loaded == manifest
    assert loaded.manifest_hash == manifest.manifest_hash


def test_global_cosine_does_not_reset_at_stage_boundary():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], lr=1.0)
    scheduler = GlobalCosineScheduler(optimizer, total_steps=10, warmup_steps=2, step=3)
    before = scheduler.global_step
    scheduler.step()
    assert scheduler.global_step == before + 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(global_cosine_factor(4, 10, 2))
    state = scheduler.state_dict()
    replacement = GlobalCosineScheduler(
        torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], lr=1.0), total_steps=10, warmup_steps=2
    )
    replacement.load_state_dict(state)
    assert replacement.global_step == scheduler.global_step


def test_parameter_group_names_cover_components():
    names = {
        "encoder.model.weight": "encoder",
        "bridge.controller.weight": "bridge",
        "decoder.grounded_copy.query.weight": "grounded_copy",
        "decoder.semantic_reader.output.weight": "semantic_read",
        "decoder.dual_readout.router.weight": "dual_readout",
        "decoder.backbone.layers.0.cross.q_proj.weight": "cross_attention",
        "decoder.lm_head.weight": "decoder",
    }
    assert {name: _component(name) for name in names} == names


def test_checkpoint_rejects_config_graph_change(tmp_path):
    config = load_config(ROOT / "configs" / "afmr_smoke.yaml")
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, config, epoch=1, step=2)
    clone = torch.nn.Linear(3, 2)
    load_checkpoint(path, clone, config=config, restore_rng=False)
    altered = load_config(ROOT / "configs" / "afmr_smoke.yaml")
    altered["decoder"]["grounded_copy"]["alpha_max"] = 0.1
    with pytest.raises(ValueError, match="architecture_spec|config_fingerprint"):
        load_checkpoint(path, clone, config=altered, restore_rng=False)


def test_resume_rejects_world_size_protocol_change(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs" / "afmr_smoke.yaml")
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, config, epoch=1, step=2)
    clone = torch.nn.Linear(3, 2)
    monkeypatch.setattr(checkpoint_module, "world_size", lambda: 2)
    with pytest.raises(ValueError, match="world_size"):
        load_checkpoint(path, clone, config=config, restore_rng=False, validate_training_protocol=True)


def test_prediction_manifest_rejects_stale_identity(tmp_path):
    output = tmp_path / "predictions.jsonl"
    identity = {"checkpoint_sha256": "a", "split": "test", "schema_version": 2}
    ensure_evaluation_manifest(output, identity)
    output.write_text('{"id":"1","prediction":"x","reference":"y"}\n', encoding="utf-8")
    ensure_evaluation_manifest(output, identity)
    with pytest.raises(ValueError, match="provenance mismatch"):
        ensure_evaluation_manifest(output, {**identity, "checkpoint_sha256": "b"})


def test_sampling_applies_temperature_top_k_and_top_p_only_to_final_scores():
    scores = torch.tensor([[4.0, 3.0, 0.0]])
    torch.manual_seed(4)
    token = _sample_token(scores, top_p=0.8, temperature=0.5, generator=None)
    assert token.shape == (1,)
    assert token.item() in {0, 1}
    assert _sample_token(scores, top_p=1.0, temperature=1.0, generator=None, top_k=1).item() == 0
    with pytest.raises(ValueError):
        _sample_token(scores, top_p=0.0, temperature=1.0, generator=None)
    with pytest.raises(ValueError):
        _sample_token(scores, top_p=1.0, temperature=1.0, generator=None, top_k=-1)


def test_tiny_model_forward_and_backward_uses_no_network():
    config = load_config(ROOT / "configs" / "afmr_smoke.yaml")
    from afmr_core.runtime import build_loaders

    loader = build_loaders(config, split="train", max_train_examples=2)["train"]
    batch = next(iter(loader))
    model = AFMRModel(config)
    copy_keys = (
        "copy_token_ids",
        "copy_token_mask",
        "copy_encoder_indices",
        "copy_token_indices",
        "copy_alignment_weights",
    )
    output = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["source_content_mask"],
        batch["decoder_prompt_ids"],
        batch["decoder_prompt_mask"],
        batch["decoder_input_ids"],
        batch["decoder_attention_mask"],
        batch["labels"],
        **{key: batch[key] for key in copy_keys},
    )
    assert output.logits.shape[-1] == 128
    assert torch.isfinite(output.loss_ce)
    output.loss_ce.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
