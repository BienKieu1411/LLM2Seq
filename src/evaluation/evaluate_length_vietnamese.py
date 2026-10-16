#!/usr/bin/env python3
"""Compute prediction/reference length metrics for the Vietnamese TXT layout.

Expected tree (the same layout consumed by ``evaluate_rouge_vietnamese.py``)::

    data_sum/<model>/<dataset>/<file>.txt
    data_ref/<dataset>/<file>.txt

The word-count and length-score formulas match ``huy_calculate_len.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PLATFORM_ROOT_CANDIDATES = (Path("/workspace/storage-shared/nlp/dungdx4/textsum_platform_eval"),)
WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", flags=re.UNICODE)
METRIC_COLUMNS = (
    "abs_len_loss",
    "length_deviation",
    "abs_length_deviation",
    "length_score",
    "generated_words",
    "target_words",
)


def _platform_root() -> Path:
    configured = os.environ.get("TEXTSUM_PLATFORM_EVAL_ROOT")
    if configured:
        return Path(configured).expanduser()
    return next((path for path in PLATFORM_ROOT_CANDIDATES if path.is_dir()), PLATFORM_ROOT_CANDIDATES[0])


@dataclass(frozen=True)
class Pair:
    model: str
    dataset: str
    relative_file: str
    prediction: Path
    reference: Path


def count_words(text: str) -> int:
    return len(WORD_RE.findall(unicodedata.normalize("NFKC", text)))


def compute_length_metrics(generated_words: int, target_words: int) -> dict[str, float]:
    if target_words <= 0:
        raise ValueError("reference/target word count must be positive")
    deviation = (generated_words - target_words) / target_words
    score = 100.0 * math.exp(5.0 * deviation if deviation < 0.0 else -2.0 * deviation)
    return {
        "abs_len_loss": float(abs(generated_words - target_words)),
        "length_deviation": float(deviation),
        "abs_length_deviation": float(abs(deviation)),
        "length_score": float(score),
        "generated_words": float(generated_words),
        "target_words": float(target_words),
    }


def _selected_models(root: Path, model_names: list[str] | None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Prediction root not found: {root}")
    if model_names:
        directories = [root / name for name in model_names]
        missing = [str(path) for path in directories if not path.is_dir()]
        if missing:
            raise FileNotFoundError("Model directories not found: " + ", ".join(missing))
        return directories
    return sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))


def discover_pairs(
    prediction_root: Path,
    reference_root: Path,
    model_names: list[str] | None,
    datasets: list[str] | None,
) -> tuple[list[Pair], list[str]]:
    if not reference_root.is_dir():
        raise FileNotFoundError(f"Reference root not found: {reference_root}")
    pairs: list[Pair] = []
    missing: list[str] = []
    selected = set(datasets or ())
    for model_dir in _selected_models(prediction_root, model_names):
        for dataset_dir in sorted(
            path for path in model_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
        ):
            if selected and dataset_dir.name not in selected:
                continue
            for prediction in sorted(dataset_dir.rglob("*.txt")):
                relative = prediction.relative_to(dataset_dir)
                if any(part.startswith(".") for part in relative.parts):
                    continue
                reference = reference_root / dataset_dir.name / relative
                if not reference.is_file():
                    missing.append(str(reference))
                    continue
                pairs.append(
                    Pair(
                        model=model_dir.name,
                        dataset=dataset_dir.name,
                        relative_file=relative.as_posix(),
                        prediction=prediction,
                        reference=reference,
                    )
                )
    return pairs, missing


def score_pair(pair: Pair) -> dict[str, Any]:
    prediction = pair.prediction.read_text(encoding="utf-8-sig").strip()
    reference = pair.reference.read_text(encoding="utf-8-sig").strip()
    metrics = compute_length_metrics(count_words(prediction), count_words(reference))
    return {"model": pair.model, "dataset": pair.dataset, "file": pair.relative_file, **metrics}


def average_rows(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    values = list(rows)
    result: dict[str, float | int] = {"count": len(values)}
    for key in METRIC_COLUMNS:
        result[key] = sum(float(row[key]) for row in values) / len(values) if values else 0.0
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    return {
        "overall": average_rows(rows),
        "datasets": {name: average_rows(dataset_rows) for name, dataset_rows in sorted(by_dataset.items())},
    }


def write_reports(prediction_root: Path, rows: list[dict[str, Any]]) -> None:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)

    columns = ["model", "dataset", "file", *METRIC_COLUMNS]
    for model, model_rows in sorted(by_model.items()):
        model_rows.sort(key=lambda row: (str(row["dataset"]), str(row["file"])))
        model_dir = prediction_root / model
        csv_path = model_dir / "length_scores.csv"
        json_path = model_dir / "length_summary.json"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(model_rows)

        report = {
            "model": model,
            "word_count": "NFKC Unicode words",
            "target": "matching reference file word count; all text in the file is counted",
            "length_deviation": "(generated_words - target_words) / target_words",
            "length_score": "100*exp(5*LD) if LD<0 else 100*exp(-2*LD)",
            "aggregation": "macro average over files",
            **summarize(model_rows),
        }
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        overall = report["overall"]
        print(f"\n{model} ({overall['count']} files)")
        print(f"  Absolute length loss: {overall['abs_len_loss']:.6f}")
        print(f"  Length deviation:     {overall['length_deviation']:.6f}")
        print(f"  Length score:         {overall['length_score']:.6f}")
        print(f"  Per-file: {csv_path}")
        print(f"  Summary:  {json_path}")


def parse_args() -> argparse.Namespace:
    platform_root = _platform_root()
    parser = argparse.ArgumentParser(
        description="Compute length metrics for Vietnamese TXT predictions and matching references.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-sum-root",
        "--output-root",
        dest="data_sum_root",
        type=Path,
        default=Path(os.environ.get("TEXTSUM_SUMMARY_ROOT", platform_root / "data_sum")),
        help="Root containing <model>/<dataset>/*.txt predictions",
    )
    parser.add_argument(
        "--data-ref-root",
        type=Path,
        default=Path(os.environ.get("TEXTSUM_REFERENCE_ROOT", platform_root / "data_ref")),
        help="Root containing <dataset>/*.txt references",
    )
    parser.add_argument(
        "--model-name",
        "--model",
        action="append",
        dest="models",
        help="Model directory name under data-sum-root; repeat to evaluate multiple models. Default: all model directories.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset subset; default: every dataset directory found under each model.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip predictions without matching reference files instead of failing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_datasets = [part.strip() for part in args.datasets.split(",") if part.strip()] if args.datasets else None
    try:
        pairs, missing = discover_pairs(
            args.data_sum_root.expanduser().resolve(),
            args.data_ref_root.expanduser().resolve(),
            args.models,
            selected_datasets,
        )
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if missing:
        print(f"Found {len(missing)} predictions without matching references.", file=sys.stderr)
        for path in missing[:20]:
            print(f"  missing: {path}", file=sys.stderr)
        if not args.allow_missing:
            print("Use --allow-missing to evaluate matched pairs only.", file=sys.stderr)
            return 2
    if not pairs:
        print("ERROR: no matching prediction/reference .txt files found", file=sys.stderr)
        return 2

    try:
        rows = [score_pair(pair) for pair in pairs]
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    write_reports(args.data_sum_root.expanduser().resolve(), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
