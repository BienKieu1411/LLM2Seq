#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eviseq_update_v3.config import load_config  # noqa: E402
from eviseq_update_v3.modeling.model import EviSeqAFMR  # noqa: E402
from eviseq_update_v3.runtime import _write_resolved_config, build_loaders, evaluate  # noqa: E402
from eviseq_update_v3.training.checkpoint import load_checkpoint  # noqa: E402
from eviseq_update_v3.training.engine import AFMRTrainer, seed_everything  # noqa: E402


def main() -> None:
    seed_everything(7)
    config = load_config(ROOT / "configs/afmr_smoke.yaml")
    config["decoder"]["query_cross_gate"] = False
    config["decoder"]["grounded_copy"]["enabled"] = True
    config["decoder"]["grounded_copy"]["semantic_read"].update(
        enabled=True,
        num_heads=4,
        rank=32,
        attention="hierarchical_coverage",
        fusion="norm_preserving",
        head_gate_position="post_norm",
        planner={"region_size": 1, "partition_heads": False},
    )
    config["generation"]["max_new_tokens"] = 4
    loaders = build_loaders(config, max_train_examples=4, max_validation_examples=2)
    model = EviSeqAFMR(config)
    batch = next(iter(loaders["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    initial = model(**tensors, return_logits=False)
    initial.loss.backward()
    head = model.decoder.grounded_copy
    for parameter in head.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError("Grounded-copy/semantic-read CE gradient is missing or non-finite")
    if not head.semantic_output.weight.grad.abs().any():
        raise RuntimeError("The zero-initialized semantic output projection cannot start learning")
    initial_loss = float(initial.loss.detach())
    del initial
    model.zero_grad(set_to_none=True)
    with tempfile.TemporaryDirectory(prefix="eviseq_update_v3_smoke_") as directory:
        output_dir = Path(directory)
        config["experiment"]["output_dir"] = str(output_dir)
        _write_resolved_config(config, output_dir)
        trainer = AFMRTrainer(model, config, "cpu")
        trainer.fit(loaders["train"], loaders["validation"])
        model.zero_grad(set_to_none=True)
        model(**tensors, return_logits=False).loss.backward()
        for parameter in head.parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all() or not parameter.grad.abs().any():
                raise RuntimeError("A trained grounded-copy/semantic-read parameter has no finite nonzero CE gradient")
        model.zero_grad(set_to_none=True)
        restored = EviSeqAFMR(config).eval()
        metadata = load_checkpoint(output_dir / "last.pt", restored, config=config)
        with torch.no_grad():
            final = restored(**tensors, return_logits=False)
            dense = restored(**tensors, return_logits=True)
            torch.testing.assert_close(final.loss, dense.loss)
        predictions = output_dir / "last_test_predictions.jsonl"
        metrics = evaluate(
            output_dir / "resolved_config.yaml", output_dir / "last.pt", predictions, split="test", device="cpu"
        )
        resumed = evaluate(
            output_dir / "resolved_config.yaml", output_dir / "last.pt", predictions, split="test", device="cpu"
        )
        if resumed != metrics:
            raise RuntimeError("Completed prediction resume changed evaluation results")
        output = {
            "status": "ok",
            "architecture": config["architecture"]["name"],
            "grounded_copy": True,
            "semantic_read": True,
            "semantic_attention": head.semantic_attention,
            "semantic_max_relative_rms": head.semantic_max_relative_rms,
            "query_cross_gate": config["decoder"]["query_cross_gate"],
            "semantic_heads": config["decoder"]["grounded_copy"]["semantic_read"]["num_heads"],
            "semantic_fusion": head.semantic_fusion,
            "semantic_head_gate_position": head.semantic_head_gate_position,
            "partition_heads": head.partition_heads,
            "initial_loss": initial_loss,
            "final_loss": float(final.loss),
            "checkpoint_epoch": metadata["epoch"],
            "checkpoint_step": metadata["step"],
            "generated_examples": metrics["num_examples"],
        }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
