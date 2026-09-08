"""Evidence cache, sparse collation, and actual-read contrastive gradients."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from eviseq_update.config import validate_config
from eviseq_update.data.collate import SummarizationCollator
from eviseq_update.data.dataset import JsonlSummarizationDataset
from eviseq_update.data.evidence import EVIDENCE_TENSOR_KEYS
from eviseq_update.data.evidence_cache import cache_config_fingerprint, write_cache
from eviseq_update.data.evidence_mining import build_annotation
from eviseq_update.modeling.model import EviSeqAFMR
from eviseq_update.runtime import _TinyTokenizer, _write_resolved_config, build_loaders
from eviseq_update.training.engine import AFMRTrainer
from test_semantic_read import model_config

SOURCE = "The intervention reduced mortality among surgical patients. Winter mortality increased in migratory birds."
TARGET = "The intervention reduced mortality among surgical patients."


def _make_cache(tmp_path, config):
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"id": "x", "text": SOURCE, "summary": TARGET}) + "\n", encoding="utf-8")
    for split in ("train", "validation", "test"):
        config["data"][f"{split}_file"] = str(data)
    dataset = JsonlSummarizationDataset(data, config["data"])
    collator = SummarizationCollator(
        _TinyTokenizer(), _TinyTokenizer(), config["data"], grounded_copy=True, evidence_mode="both"
    )
    annotation, skips = build_annotation(0, dataset[0], collator)
    assert not skips.get("insufficient_positive_or_negative")
    assert annotation.units
    cache = write_cache(tmp_path / "evidence", [annotation], {"config_fingerprint": cache_config_fingerprint(config)})
    return cache, annotation


def _config(tmp_path):
    config = model_config("independent_bounded")
    config["training"].update(
        interface_warmup_epochs=0, full_finetune_epochs=1, batch_size=1, gradient_accumulation_steps=1
    )
    config["training"]["evidence_contrastive"] = {
        "enabled": True,
        "cache_path": str(tmp_path / "evidence" / "evidence.jsonl"),
        "mode": "both",
        "max_weight": 0.05,
        "ramp_ratio": 0.10,
        "mining": {},
    }
    return config


def test_mining_cache_and_sparse_batch_preserve_the_lexical_ambiguity(tmp_path):
    config = _config(tmp_path)
    cache, annotation = _make_cache(tmp_path, config)
    validate_config(config)
    mortality = next(unit for unit in annotation.units if unit.target_positions)
    assert mortality.copy_positive_by_target and mortality.copy_negative_by_target
    assert mortality.semantic_positive_spans and mortality.semantic_negative_spans
    batch = next(iter(build_loaders(config)["train"]))
    assert batch["evidence_unit_valid"].any()
    assert batch["evidence_copy_positive"].any() and (~batch["evidence_copy_positive"]).any()
    assert batch["evidence_semantic_positive"].any() and (~batch["evidence_semantic_positive"]).any()
    assert cache.is_file()


def test_actual_copy_and_semantic_read_logits_receive_contrastive_gradient(tmp_path):
    torch.manual_seed(123)
    config = _config(tmp_path)
    _make_cache(tmp_path, config)
    batch = next(iter(build_loaders(config)["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    model = EviSeqAFMR(config).train()
    no_evidence = {key: value for key, value in tensors.items() if key not in EVIDENCE_TENSOR_KEYS}
    baseline = model(**no_evidence, return_logits=False)
    output = model(**tensors, return_logits=False)
    torch.testing.assert_close(output.loss_ce, baseline.loss_ce, atol=0, rtol=0)
    statistics = output.loss_statistics
    assert statistics is not None
    assert statistics.copy_sum.item() > 0 and statistics.semantic_sum.item() > 0
    (statistics.copy_sum + statistics.semantic_sum).backward()
    head = model.decoder.grounded_copy
    for parameter in (
        head.query.weight,
        head.context_key.weight,
        head.lexical_key.weight,
        head.semantic_query.weight,
        head.semantic_key.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_cache_rejects_changed_normalized_source(tmp_path):
    config = _config(tmp_path)
    cache, _ = _make_cache(tmp_path, config)
    changed = tmp_path / "changed.jsonl"
    changed.write_text(
        json.dumps({"id": "x", "text": SOURCE + " Changed.", "summary": TARGET}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="content hash mismatch"):
        JsonlSummarizationDataset(changed, config["data"], evidence_cache_path=cache)[0]


def test_prepare_evidence_cli_writes_a_full_sidecar(tmp_path):
    config = _config(tmp_path)
    data = tmp_path / "train.jsonl"
    data.write_text(
        "".join(json.dumps({"id": str(index), "text": SOURCE, "summary": TARGET}) + "\n" for index in range(2)),
        encoding="utf-8",
    )
    for split in ("train", "validation", "test"):
        config["data"][f"{split}_file"] = str(data)
    _write_resolved_config(config, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "prepare_evidence.py"),
            "--config",
            str(tmp_path / "resolved_config.yaml"),
            "--split",
            "train",
            "--output-dir",
            str(tmp_path / "evidence"),
            "--audit-size",
            "1",
        ],
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = (tmp_path / "evidence" / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert len((tmp_path / "evidence" / "audit_examples.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads((tmp_path / "evidence" / "summary.json").read_text())["records"] == 2


def test_trainer_normalizes_and_backpropagates_the_contrastive_sums(tmp_path):
    torch.manual_seed(19)
    config = _config(tmp_path)
    _make_cache(tmp_path, config)
    config["experiment"]["output_dir"] = str(tmp_path / "run")
    config["training"].update(log_every_steps=1, save_each_epoch=False)
    config["training"]["evidence_contrastive"]["ramp_ratio"] = 0.0
    loaders = build_loaders(config)
    model = EviSeqAFMR(config)
    before = model.decoder.grounded_copy.query.weight.detach().clone()
    AFMRTrainer(model, config, "cpu").fit(loaders["train"])
    assert not torch.equal(before, model.decoder.grounded_copy.query.weight)
    rows = [json.loads(line) for line in (tmp_path / "run" / "training_metrics.jsonl").read_text().splitlines()]
    step = next(row for row in rows if row["type"] == "step")
    assert step["evidence_lambda"] == 0.05 and step["evidence_units"] > 0


def _distributed_worker(config_path: str) -> None:
    from eviseq_update.runtime import train

    torch.set_num_threads(1)
    train(config_path, device="cpu")


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Requires Gloo")
def test_ddp_masks_evidence_for_the_final_placeholder_rank(tmp_path):
    config = _config(tmp_path)
    data = tmp_path / "train.jsonl"
    data.write_text(
        "".join(json.dumps({"id": str(index), "text": SOURCE, "summary": TARGET}) + "\n" for index in range(3)),
        encoding="utf-8",
    )
    for split in ("train", "validation", "test"):
        config["data"][f"{split}_file"] = str(data)
    plain = JsonlSummarizationDataset(data, config["data"])
    collator = SummarizationCollator(_TinyTokenizer(), _TinyTokenizer(), config["data"], grounded_copy=True)
    annotations = [build_annotation(index, plain[index], collator)[0] for index in range(len(plain))]
    cache = write_cache(tmp_path / "evidence", annotations, {"config_fingerprint": cache_config_fingerprint(config)})
    config["training"].update(
        interface_warmup_epochs=0,
        full_finetune_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        num_workers=0,
        validation_num_workers=0,
        length_bucketing=False,
        log_every_steps=1,
        save_each_epoch=False,
    )
    config["training"]["evidence_contrastive"].update(cache_path=str(cache), ramp_ratio=0.0)
    config["experiment"]["output_dir"] = str(tmp_path / "run")
    _write_resolved_config(config, tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "--nproc_per_node=2",
            str(Path(__file__).resolve()),
            "--evidence-worker",
            str(tmp_path / "resolved_config.yaml"),
        ],
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).parents[1]),
            OMP_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        ),
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in (tmp_path / "run" / "training_metrics.jsonl").read_text().splitlines()]
    steps = [row for row in rows if row["type"] == "step"]
    assert [row["examples"] for row in steps] == [2, 1]
    assert [row["evidence_units"] for row in steps] == [2, 1]


if __name__ == "__main__":
    if sys.argv[1:2] == ["--evidence-worker"]:
        _distributed_worker(sys.argv[2])
