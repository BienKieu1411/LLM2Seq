"""Run the controlled GenBridge ablation matrix sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import MODEL_PROFILES, apply_model_size, load_config

GROUPS = {
    # Fast decision gate before committing the 1.7B paper run.
    "pilot": ["direct_qwen", "causal_ed", "lamate_style", "concat_memory", "genbridge"],
    # Minimal table needed to defend the architecture against LaMaTE.
    "main": [
        "direct_qwen",
        "causal_ed",
        "lamate_style",
        "hierarchical",
        "no_salience_loss",
        "no_memory_curriculum",
        "no_salience_attention_bias",
        "no_plan_evidence_alignment",
        "plan_only",
        "concat_memory",
        "genbridge",
    ],
    "analysis": [
        "adapter_2layers",
        "adapter_8layers",
        "no_adapter_rope",
        "plan_only",
        "concat_memory",
        "layer_fusion",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=[*GROUPS, "all"], default="pilot")
    parser.add_argument("--model-size", choices=sorted(MODEL_PROFILES), default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = sorted({name for group in GROUPS.values() for name in group}) if args.group == "all" else GROUPS[args.group]
    for name in names:
        config = root / "configs" / "ablations" / f"{name}.yaml"
        resolved = load_config(config)
        apply_model_size(resolved, args.model_size)
        output_dir = Path(str(resolved["experiment"]["output_dir"]))
        train_command = [
            sys.executable,
            "-m",
            "genbridge.training",
            "--config",
            str(config),
        ]
        if args.model_size:
            train_command.extend(["--model-size", args.model_size])
        if args.overwrite_output_dir:
            train_command.append("--overwrite-output-dir")
        if not args.eval_only:
            print(" ".join(train_command), flush=True)
            if not args.dry_run:
                result = subprocess.run(train_command, check=False)
                if result.returncode:
                    if args.continue_on_error:
                        continue
                    raise SystemExit(result.returncode)
        if args.evaluate or args.eval_only:
            eval_command = [
                sys.executable,
                "-m",
                "genbridge.evaluate",
                "--config",
                str(output_dir / "resolved_config.yaml"),
                "--checkpoint",
                str(output_dir / "best.pt"),
                "--output",
                str(output_dir / "test_predictions.jsonl"),
            ]
            print(" ".join(eval_command), flush=True)
            if not args.dry_run:
                result = subprocess.run(eval_command, check=False)
                if result.returncode and not args.continue_on_error:
                    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
