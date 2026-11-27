#!/usr/bin/env python3
"""Print the settings that must match before runs are compared across seeds. Read-only.

Usage: python3 audit_runs.py RUN_DIR [RUN_DIR ...]
Every directory is searched for resolved_config.yaml. For each run found this prints what was
configured and what actually happened: effective batch inferred from optimizer steps per epoch,
the learning rate observed a quarter of the way through full fine-tuning (linear decay gives
0.75 of the base rate, cosine 0.85), validation loss per epoch and test prediction rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

LR_KEYS = ("full_encoder_lr", "full_decoder_lr", "full_bridge_lr", "full_cross_attention_lr")


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def audit(run: Path) -> list[str]:
    config = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    training, decoder = config["training"], config["decoder"]
    copy = (decoder.get("grounded_copy") or decoder.get("source_copy") or {}).get("enabled")
    warmup, full = int(training["interface_warmup_epochs"]), int(training["full_finetune_epochs"])
    steps, validation, elapsed = [], {}, None
    metrics = run / "training_metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            elapsed = row.get("total_elapsed_seconds", elapsed)
            if row.get("type") == "step" and row.get("stage") == "full_finetune":
                steps.append(row)
            elif row.get("type") == "epoch" and row.get("split") == "validation":
                validation[row["epoch"]] = round(row["ce"], 4)
    lines = [
        str(run),
        f"  seed={training['seed']} bridge={config['architecture']['bridge_mode']} copy={copy}",
        f"  warmup_epochs={warmup} full_epochs={full} batch={training['batch_size']} "
        f"accum={training['gradient_accumulation_steps']} "
        f"lr_scheduler={training.get('lr_scheduler', 'not in config')}",
    ]
    if steps:
        per_epoch = steps[-1]["epoch_steps"]
        train_file = Path(config["data"]["train_file"])
        effective = round(count_lines(train_file) / per_epoch) if train_file.is_file() else "unknown (train file not found)"
        lines.append(f"  optimizer_steps_per_epoch={per_epoch} effective_batch~{effective}")

        def progress(row: dict) -> float:
            return (row["schedule_step"] - warmup * per_epoch) / (full * per_epoch)

        quarter = min(steps, key=lambda row: abs(progress(row) - 0.25))
        base = max(float(training[key]) for key in LR_KEYS if key in training)
        ratio = max(quarter["learning_rate"].values()) / base
        lines.append(f"  lr_at_{progress(quarter):.0%}_of_full_stage={ratio:.2f} x base (linear 0.75, cosine 0.85)")
    else:
        lines.append("  no full fine-tuning step records in training_metrics.jsonl")
    lines.append(f"  validation_ce_by_epoch={[validation[epoch] for epoch in sorted(validation)]}")
    if elapsed is not None:
        lines.append(f"  train_hours={elapsed / 3600:.1f} (training and validation; test evaluation not included)")
    predictions = sorted(run.glob("*test_predictions.jsonl"))
    for path in predictions:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
        ids = {str(row.get("id", index)) for index, row in enumerate(rows)}
        lines.append(f"  {path.name}: rows={len(rows)} unique_ids={len(ids)}")
    if not predictions:
        lines.append("  no *test_predictions.jsonl in this run directory")
    return lines


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    runs = sorted({found.parent for root in argv for found in Path(root).rglob("resolved_config.yaml")})
    if not runs:
        print("No resolved_config.yaml found under the given directories", file=sys.stderr)
        return 1
    for run in runs:
        print("\n".join(audit(run)), end="\n\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
