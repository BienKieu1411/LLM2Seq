"""Train GenBridge, then evaluate both validation-best and terminal checkpoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .config import MODEL_PROFILES, apply_model_size, load_config


def pipeline_commands(
    config_path: Path,
    model_size: str | None,
    overwrite_output_dir: bool,
    eval_only: bool,
    max_samples: int | None,
) -> tuple[Path, list[list[str]]]:
    """Build deterministic child commands using the current Python runtime."""

    config = load_config(config_path)
    apply_model_size(config, model_size)
    output_dir = Path(str(config["experiment"]["output_dir"]))
    commands: list[list[str]] = []
    if not eval_only:
        train_command = [
            sys.executable,
            "-m",
            "genbridge.training",
            "--config",
            str(config_path),
        ]
        if model_size:
            train_command.extend(["--model-size", model_size])
        if overwrite_output_dir:
            train_command.append("--overwrite-output-dir")
        commands.append(train_command)

    resolved_config = output_dir / "resolved_config.yaml"
    for role in ("best", "last"):
        eval_command = [
            sys.executable,
            "-m",
            "genbridge.evaluate",
            "--config",
            str(resolved_config),
            "--checkpoint",
            str(output_dir / f"{role}.pt"),
            "--output",
            str(output_dir / f"{role}_test_predictions.jsonl"),
        ]
        if max_samples is not None:
            eval_command.extend(["--max-samples", str(max_samples)])
        commands.append(eval_command)
    return output_dir, commands


def run_commands(commands: Sequence[Sequence[str]], dry_run: bool = False) -> None:
    for command in commands:
        print(" ".join(str(part) for part in command), flush=True)
        if not dry_run:
            subprocess.run(list(command), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-size", choices=sorted(MODEL_PROFILES), default=None)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output_dir, commands = pipeline_commands(
        Path(args.config).expanduser().resolve(),
        args.model_size,
        args.overwrite_output_dir,
        args.eval_only,
        args.max_samples,
    )
    run_commands(commands, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Pipeline complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
