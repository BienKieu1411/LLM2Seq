#!/usr/bin/env python3
"""Download bltlab/lr-sum Vietnamese and convert it to GenBridge JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bltlab/lr-sum")
    parser.add_argument("--language", default="vie")
    parser.add_argument("--output-dir", type=Path, default=Path("data/lrsum"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--include-title", action="store_true")
    args = parser.parse_args()

    import polars as pl
    from huggingface_hub import hf_hub_download

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        filename = f"{args.language}/{split}-00000-of-00001.parquet"
        parquet = hf_hub_download(
            repo_id=args.dataset,
            filename=filename,
            repo_type="dataset",
            cache_dir=args.cache_dir,
        )
        output = args.output_dir / f"{split}.jsonl"
        kept = 0
        with output.open("w", encoding="utf-8") as handle:
            for row in pl.read_parquet(parquet).iter_rows(named=True):
                title = str(row.get("title") or "").strip()
                text = str(row.get("text") or "").strip()
                target = str(row.get("summary") or "").strip()
                if not text or not target:
                    continue
                parts = [f"Tiêu đề: {title}"] if args.include_title and title else []
                parts.append(text)
                example = {
                    "id": str(row.get("id") or f"{split}_{kept:06d}"),
                    "source": "\n\n".join(parts),
                    "target": target,
                    "task": "summarization",
                }
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                kept += 1
        print(f"{split}: {kept} examples -> {output}")


if __name__ == "__main__":
    main()
