"""Run the controlled AdaBiMask ablation matrix sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

GROUPS = {
    "pilot": ["causal", "full", "bottom_k8", "middle_k8", "top_k8"],
    "main": ["direct_qwen", "causal", "full", "middle_k8", "random_k8", "learnable_k8"],
    "budget": ["learnable_k4", "learnable_k8", "learnable_k12"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["pilot", "main", "budget", "all"], default="pilot")
    parser.add_argument("--warmup-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    names = (
        sorted({name for values in GROUPS.values() for name in values}) if args.group == "all" else GROUPS[args.group]
    )
    for name in names:
        config = root / "configs" / "ablations" / f"{name}.yaml"
        command = [sys.executable, "-m", "adabimask.training", "--config", str(config)]
        if args.warmup_checkpoint and name != "direct_qwen":
            command.extend(["--resume", args.warmup_checkpoint])
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, check=False)
        if result.returncode and not args.continue_on_error:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
