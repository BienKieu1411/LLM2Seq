"""Evaluate a trained EviSeq-KD checkpoint on validation or test JSONL."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from .model import EviSeqKD
from .student.configuration import load_config
from .student.evaluation import engine as stable


def _evaluation_config(path: str, split: str) -> Dict[str, Any]:
    config = load_config(path)
    if split == "validation":
        config["data"]["test_file"] = config["data"]["validation_file"]
        config.setdefault("limits", {})["max_test_examples"] = int(
            config.get("limits", {}).get("max_validation_examples", 0)
        )
    elif not str(config["data"].get("test_file", "")).strip():
        raise ValueError("The config has no data.test_file; evaluate validation or configure a test file")
    return config


def evaluate(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    *,
    split: str = "test",
    max_samples: int = 0,
    batch_size: int = 0,
) -> Dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    config = _evaluation_config(config_path, split)
    original_loader = stable.load_config
    original_model = stable.RuntimeModel
    stable.load_config = lambda _: config
    stable.RuntimeModel = EviSeqKD
    previous_batch_override = os.environ.get("EVISEQ_EVAL_BATCH_SIZE")
    if int(batch_size) > 0:
        os.environ["EVISEQ_EVAL_BATCH_SIZE"] = str(int(batch_size))
    try:
        metrics = stable.evaluate(config_path, checkpoint_path, output_path, max_samples)
    finally:
        stable.load_config = original_loader
        stable.RuntimeModel = original_model
        if previous_batch_override is None:
            os.environ.pop("EVISEQ_EVAL_BATCH_SIZE", None)
        else:
            os.environ["EVISEQ_EVAL_BATCH_SIZE"] = previous_batch_override
    metrics["evaluation_split"] = split
    metrics_path = os.path.splitext(output_path)[0] + ".metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an EviSeq-KD checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()
    evaluate(
        args.config,
        args.checkpoint,
        args.output,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
