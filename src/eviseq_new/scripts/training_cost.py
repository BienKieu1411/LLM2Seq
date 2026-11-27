"""Training cost (GPU-hours, peak VRAM, throughput) read from logs that already exist.

SEAM writes ``training_metrics.jsonl``; the decoder-only runs write ``run_manifest.json``;
T5Gemma2 and LED have no cost file, so their numbers are passed by hand. Both files record
wall-clock training time, so give ``--gpus LABEL=N`` for a run trained on N GPUs: GPU-hours is
wall hours times N. Peak VRAM and tokens/s are left as logged. Test-time generation is not included.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Sequence


def seam_training_cost(metrics_jsonl: Path) -> dict[str, float | None]:
    rows = [json.loads(line) for line in Path(metrics_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    steps = [row for row in rows if row.get("type") == "step"]
    if not steps:
        raise ValueError(f"No step records in {metrics_jsonl}")
    vram = [row["peak_vram_gib"] for row in steps if row.get("peak_vram_gib") is not None]
    throughput = [row["tokens_per_second"] for row in steps if row.get("tokens_per_second") is not None]
    return {
        "gpu_hours": float(steps[-1]["total_elapsed_seconds"]) / 3600.0,
        "peak_vram_gib": float(max(vram)) if vram else None,
        "tokens_per_second": float(fmean(throughput)) if throughput else None,
    }


def manifest_training_cost(run_manifest: Path) -> dict[str, float | None]:
    manifest = json.loads(Path(run_manifest).read_text(encoding="utf-8"))
    return {"gpu_hours": float(manifest["elapsed_seconds"]) / 3600.0, "peak_vram_gib": None, "tokens_per_second": None}


def _labelled(values: Sequence[str] | None, option: str) -> list[tuple[str, str]]:
    pairs = []
    for value in values or []:
        label, separator, rest = value.partition("=")
        if not separator or not label or not rest:
            raise ValueError(f"{option} expects LABEL=VALUE, got {value!r}")
        pairs.append((label, rest))
    return pairs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seam", action="append", help="LABEL=path/to/training_metrics.jsonl")
    parser.add_argument("--manifest", action="append", help="LABEL=path/to/run_manifest.json")
    parser.add_argument("--manual", action="append", help="LABEL=GPU_HOURS:PEAK_GIB (T5Gemma2, LED)")
    parser.add_argument("--gpus", action="append", help="LABEL=N for a --seam or --manifest run trained on N GPUs")
    args = parser.parse_args(argv)
    gpus = {label: int(value) for label, value in _labelled(args.gpus, "--gpus")}
    rows: list[tuple[str, dict[str, float | None]]] = []
    for label, path in _labelled(args.seam, "--seam"):
        rows.append((label, seam_training_cost(Path(path))))
    for label, path in _labelled(args.manifest, "--manifest"):
        rows.append((label, manifest_training_cost(Path(path))))
    unknown = sorted(set(gpus) - {label for label, _ in rows})
    if unknown:
        raise ValueError(f"--gpus applies only to --seam and --manifest labels, not {', '.join(unknown)}")
    for label, cost in rows:
        count = gpus.get(label, 1)
        cost.update(gpus=count, wall_hours=cost["gpu_hours"], gpu_hours=cost["gpu_hours"] * count)
    for label, value in _labelled(args.manual, "--manual"):
        hours, _, peak = value.partition(":")
        rows.append((label, {"gpu_hours": float(hours), "peak_vram_gib": float(peak) if peak else None, "tokens_per_second": None}))
    writer = csv.writer(sys.stdout, lineterminator="\n")
    measured = ("gpu_hours", "peak_vram_gib", "tokens_per_second", "wall_hours")
    writer.writerow(["label", *measured, "gpus"])
    for label, cost in rows:
        values = ["" if cost.get(key) is None else f"{cost[key]:.4f}" for key in measured]
        writer.writerow([label, *values, cost.get("gpus", "")])


if __name__ == "__main__":
    main()
