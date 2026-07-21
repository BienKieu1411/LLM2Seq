"""Run the controlled EviBridge ablation matrix sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import QWEN35_MODEL_SIZES


GROUPS = {
    # Fast decision gate before committing the 2B paper run.
    "pilot": ["direct_qwen", "causal_ed", "lamate_style", "evibridge"],
    # Minimal table needed to defend the architecture against LaMaTE.
    "main": ["direct_qwen", "causal_ed", "lamate_style", "hierarchical", "no_evidence_loss", "slots_only", "evibridge"],
    "analysis": ["slots_only", "layer_fusion"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=[*GROUPS, "all"], default="pilot")
    parser.add_argument("--model-size", choices=sorted(QWEN35_MODEL_SIZES), default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = sorted({name for group in GROUPS.values() for name in group}) if args.group == "all" else GROUPS[args.group]
    for name in names:
        config = root / "configs" / "ablations" / f"{name}.yaml"
        command = [sys.executable, "-m", "evibridge.training", "--config", str(config)]
        if args.model_size:
            command.extend(["--model-size", args.model_size])
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, check=False)
        if result.returncode and not args.continue_on_error:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
