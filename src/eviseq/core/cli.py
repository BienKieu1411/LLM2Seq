"""Command-line interface for reusable EviSeq experiments."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

from .configuration import load_config, resolve_data_path
from .data.dataset import read_jsonl


def _add_train(subparsers: Any) -> None:
    parser = subparsers.add_parser("train", help="Fine-tune EviSeq on a configured text-to-text task")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--init-checkpoint", default="", help="Initialize a new run from an existing checkpoint")
    parser.add_argument("--allow-partial-init", action="store_true")
    parser.add_argument("--output-dir", default="", help="Override experiment.output_dir")


def _add_evaluate(subparsers: Any) -> None:
    parser = subparsers.add_parser("evaluate", help="Generate predictions and compute configured metrics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from completed JSONL predictions at --output")


def _dataset_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    data = config["data"]
    result: Dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        configured_path = str(data.get(f"{split}_file", "")).strip()
        if not configured_path:
            continue
        path = resolve_data_path(configured_path, config)
        rows = read_jsonl(path, data_config=data)
        result[split] = {
            "path": str(path),
            "examples": len(rows),
            "source_characters_mean": round(sum(len(row["source"]) for row in rows) / len(rows), 2),
            "target_characters_mean": round(sum(len(row["target"]) for row in rows) / len(rows), 2),
        }
    return result


def main() -> None:
    # Evaluation and training emit useful per-batch progress at INFO level.
    # Configure the source runner explicitly because Python's default logging
    # threshold is WARNING, which otherwise hides the ETA/progress messages.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    parser = argparse.ArgumentParser(prog="eviseq")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train(subparsers)
    _add_evaluate(subparsers)
    inspect_parser = subparsers.add_parser("inspect", help="Print the resolved configuration")
    inspect_parser.add_argument("--config", required=True)
    validate_parser = subparsers.add_parser("validate-data", help="Validate field mapping for all JSONL splits")
    validate_parser.add_argument("--config", required=True)
    args = parser.parse_args()

    if args.command == "train":
        from .training.trainer import train

        train(
            args.config,
            args.overwrite_output_dir,
            args.init_checkpoint,
            args.output_dir,
            args.allow_partial_init,
        )
    elif args.command == "evaluate":
        from .evaluation.evaluator import evaluate

        evaluate(
            args.config,
            args.checkpoint,
            args.output,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            resume=args.resume,
        )
    elif args.command == "inspect":
        print(json.dumps(load_config(args.config), ensure_ascii=False, indent=2))
    elif args.command == "validate-data":
        print(json.dumps(_dataset_summary(load_config(args.config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
