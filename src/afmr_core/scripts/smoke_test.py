#!/usr/bin/env python3
"""Offline AFMR integration smoke using only the bundled ``__tiny__`` models."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from afmr_core.config import load_config
from afmr_core.modeling.model import AFMRModel
from afmr_core.runtime import _write_resolved_config, build_loaders, evaluate
from afmr_core.training.checkpoint import load_checkpoint
from afmr_core.training.engine import AFMRTrainer, seed_everything


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    seed_everything(7)
    config = load_config(root / "configs" / "afmr_smoke.yaml")
    config["generation"]["max_new_tokens"] = 4
    loaders = build_loaders(config, max_train_examples=4, max_validation_examples=2)
    model = AFMRModel(config)
    batch = next(iter(loaders["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    initial = model(**tensors, return_logits=False)
    if initial.loss is None or not torch.isfinite(initial.loss):
        raise RuntimeError("AFMR tiny loss is not finite")
    initial.loss.backward()
    if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise RuntimeError("AFMR tiny backward produced no finite gradient")
    model.zero_grad(set_to_none=True)

    with tempfile.TemporaryDirectory(prefix="afmr_core_smoke_") as directory:
        output_dir = Path(directory)
        config["experiment"]["output_dir"] = str(output_dir)
        _write_resolved_config(config, output_dir)
        trainer = AFMRTrainer(model, config, "cpu")
        trainer.fit(loaders["train"], loaders["validation"])
        if model.decoder.semantic_reader is not None and not model.decoder.semantic_reader._tiny_calibrated:
            raise RuntimeError("tiny semantic reader was not calibrated before training")
        train_metrics = [
            json.loads(line)
            for line in (output_dir / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        train_ce = [
            float(record["ce"])
            for record in train_metrics
            if record.get("type") == "epoch" and record.get("split") == "train"
        ]
        if len(train_ce) >= 2 and not train_ce[-1] < train_ce[0]:
            raise RuntimeError(f"fixed tiny-set CE did not decrease: {train_ce}")
        restored = AFMRModel(config).eval()
        metadata = load_checkpoint(output_dir / "last.pt", restored, config=config, restore_rng=False)
        resumed_trainer = AFMRTrainer(AFMRModel(config), config, "cpu")
        resumed_trainer.fit(loaders["train"], loaders["validation"], resume_checkpoint=str(output_dir / "last.pt"))
        with torch.no_grad():
            dense = restored(**tensors, return_logits=True)
            chunked = restored(**tensors, return_logits=False)
        torch.testing.assert_close(dense.loss, chunked.loss, rtol=1e-5, atol=1e-5)

        predictions = output_dir / "test_predictions.jsonl"
        metrics = evaluate(
            output_dir / "resolved_config.yaml", output_dir / "last.pt", predictions, split="test", device="cpu"
        )
        resumed = evaluate(
            output_dir / "resolved_config.yaml", output_dir / "last.pt", predictions, split="test", device="cpu"
        )
        if metrics != resumed or metadata["step"] <= 0:
            raise RuntimeError("AFMR checkpoint/evaluation resume is not deterministic")
    print(
        json.dumps(
            {
                "status": "ok",
                "checkpoint_step": metadata["step"],
                "examples": metrics["num_examples"],
                "train_ce": train_ce,
            }
        )
    )


if __name__ == "__main__":
    main()
