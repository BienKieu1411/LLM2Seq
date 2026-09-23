#!/usr/bin/env python3
"""Merge ordered XOV evaluation shards into one prediction JSONL."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def merge(shard_paths: list[Path], output_path: Path) -> dict:
    rows_by_index: dict[int, dict] = {}
    for shard_path in shard_paths:
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing evaluation shard: {shard_path}")
        with shard_path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {shard_path}:{line_number}") from exc
                if not isinstance(row, dict) or "index" not in row:
                    raise ValueError(f"Shard row lacks an integer index: {shard_path}:{line_number}")
                try:
                    index = int(row["index"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Shard index is not an integer: {shard_path}:{line_number}") from exc
                if index < 0 or index in rows_by_index:
                    raise ValueError(f"Duplicate or invalid evaluation shard index {index}")
                if not isinstance(row.get("id"), str) or not isinstance(row.get("prediction"), str):
                    raise ValueError(f"Shard row has invalid id/prediction: {shard_path}:{line_number}")
                rows_by_index[index] = row

    if not rows_by_index:
        raise ValueError("Evaluation shards contain no rows")
    ordered_indices = sorted(rows_by_index)
    expected_indices = list(range(len(ordered_indices)))
    if ordered_indices != expected_indices:
        raise ValueError(
            "Evaluation shards are incomplete or non-contiguous: "
            f"expected indices 0..{len(ordered_indices) - 1}, got {ordered_indices[:5]}..."
        )

    ordered_rows = []
    seen_ids: set[str] = set()
    for index in ordered_indices:
        row = rows_by_index[index]
        example_id = row["id"]
        if example_id in seen_ids:
            raise ValueError(f"Duplicate evaluation example id after merge: {example_id!r}")
        seen_ids.add(example_id)
        ordered_rows.append({key: value for key, value in row.items() if key != "index"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in ordered_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, output_path)

    from xov.evaluation.metrics import summarization_metrics

    predictions = [row["prediction"] for row in ordered_rows]
    references = [row["reference"] for row in ordered_rows]
    metrics = summarization_metrics(predictions, references)
    metrics_path = Path(str(output_path) + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    metrics = merge(args.shards, args.output)
    print(
        "Merged %d rows -> %s | ROUGE-1=%.3f | ROUGE-2=%.3f | ROUGE-L=%.3f"
        % (metrics["num_examples"], args.output, metrics["rouge1"], metrics["rouge2"], metrics["rougeL"])
    )


if __name__ == "__main__":
    main()
