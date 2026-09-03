"""Reusable validation/test evaluation for EviSeq."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from ..config import load_config
from . import engine as stable


def _evaluation_config(config_path: str, split: str) -> Dict[str, Any]:
    config = load_config(config_path)
    if split == "validation":
        config["data"]["test_file"] = config["data"]["validation_file"]
        config.setdefault("limits", {})["max_test_examples"] = int(
            config.get("limits", {}).get("max_validation_examples", 0)
        )
    elif not str(config["data"].get("test_file", "")).strip():
        raise ValueError("The task has no data.test_file; evaluate the validation split or configure a test file")
    return config


def evaluate(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    *,
    split: str = "validation",
    max_samples: int = 0,
    batch_size: int = 0,
    resume: bool = False,
) -> Dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")

    config = _evaluation_config(config_path, split)
    original_loader = stable.load_config
    stable.load_config = lambda _: config
    previous_batch_override = os.environ.get("EVISEQ_EVAL_BATCH_SIZE")
    if int(batch_size) > 0:
        os.environ["EVISEQ_EVAL_BATCH_SIZE"] = str(int(batch_size))
    try:
        metrics = stable.evaluate(config_path, checkpoint_path, output_path, max_samples, resume=resume)
    finally:
        stable.load_config = original_loader
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from completed JSONL predictions at --output")
    args = parser.parse_args()
    evaluate(
        args.config,
        args.checkpoint,
        args.output,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
