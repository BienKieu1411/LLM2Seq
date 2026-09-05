"""Minimal AFMR command line entry points."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .data.prepare import prepare_split


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(prog="eviseq-afmr")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--python", default=sys.executable)
    train = subparsers.add_parser("train")
    train.add_argument("config")
    train.add_argument("--device", default=None)
    train.add_argument("--resume-checkpoint", default=None)
    train.add_argument("--max-train-examples", type=int, default=0)
    train.add_argument("--max-validation-examples", type=int, default=0)
    train.add_argument("--overwrite-output-dir", action="store_true")
    train.add_argument("--output-dir", default=None)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("config")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("output")
    evaluate.add_argument("--split", default="test", choices=("train", "validation", "test"))
    evaluate.add_argument("--device", default=None)
    evaluate.add_argument("--batch-size", type=int, default=None)
    evaluate.add_argument("--max-examples", type=int, default=0)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("input_path")
    prepare.add_argument("output_path")
    prepare_dataset = subparsers.add_parser("prepare-dataset")
    prepare_dataset.add_argument(
        "--dataset", required=True, choices=("pubmed", "arxiv", "cnndm", "wikilingua", "custom")
    )
    prepare_dataset.add_argument("--input-dir", required=True)
    prepare_dataset.add_argument("--output-dir", required=True)
    prepare_dataset.add_argument("--raw-copy-dir", default=None)
    prepare_dataset.add_argument("--allow-cross-split-content", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        script = Path(__file__).resolve().parents[1] / "scripts" / "smoke_test.py"
        raise SystemExit(subprocess.call([args.python, str(script)]))
    if args.command == "validate-config":
        config = load_config(args.config)
        print(f"valid AFMR config: {config['_meta']['config_path']}")
        return
    if args.command == "train":
        from .runtime import train as run_train

        run_train(
            args.config,
            device=args.device,
            resume_checkpoint=args.resume_checkpoint,
            max_train_examples=args.max_train_examples,
            max_validation_examples=args.max_validation_examples,
            overwrite_output_dir=args.overwrite_output_dir,
            output_dir_override=args.output_dir,
        )
        return
    if args.command == "evaluate":
        from .runtime import evaluate as run_evaluate

        result = run_evaluate(
            args.config,
            args.checkpoint,
            args.output,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            max_examples=args.max_examples,
        )
        print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "prepare-dataset":
        from .data.prepare_dataset import prepare_dataset as run_prepare_dataset

        report = run_prepare_dataset(
            args.input_dir,
            args.output_dir,
            dataset=args.dataset,
            raw_copy_dir=args.raw_copy_dir,
            allow_cross_split_content=args.allow_cross_split_content,
        )
        for split, stats in report["splits"].items():
            print(f"{split}: {stats['kept']} examples (skipped {stats['skipped']}) -> {stats['processed_path']}")
        return
    count = prepare_split(
        args.input_path,
        args.output_path,
    )
    print(f"prepared {count} records -> {args.output_path}")


if __name__ == "__main__":
    main()
