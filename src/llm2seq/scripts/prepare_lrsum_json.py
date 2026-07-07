#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

DEFAULT_DATASET = "bltlab/lr-sum"
DEFAULT_LANGUAGE = "vie"
SPLITS = ("train", "validation", "test")


def rows_with_datasets(dataset_name: str, language: str, cache_dir: str | None) -> dict[str, Iterable[dict]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, language, cache_dir=cache_dir)
    return {split: ds[split] for split in SPLITS}


def rows_with_parquet(dataset_name: str, language: str, cache_dir: str | None) -> dict[str, list[dict]]:
    import polars as pl
    from huggingface_hub import hf_hub_download

    temp_dir = Path(tempfile.mkdtemp(prefix="lrsum_dataset_"))
    cache_path = Path(cache_dir) if cache_dir else temp_dir / "hf_cache"
    try:
        by_split: dict[str, list[dict]] = {}
        for split in SPLITS:
            file = f"{language}/{split}-00000-of-00001.parquet"
            local = hf_hub_download(
                repo_id=dataset_name,
                filename=file,
                repo_type="dataset",
                cache_dir=str(cache_path),
                local_dir=str(temp_dir),
            )
            by_split[split] = pl.read_parquet(local).to_dicts()
        return by_split
    finally:
        if cache_dir is None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def load_rows(dataset_name: str, language: str, cache_dir: str | None) -> dict[str, Iterable[dict]]:
    try:
        return rows_with_datasets(dataset_name, language, cache_dir)
    except (ModuleNotFoundError, ValueError):
        return rows_with_parquet(dataset_name, language, cache_dir)


def convert_split(
    rows: Iterable[dict],
    output_path: Path,
    split_name: str,
    task_name: str,
    max_samples: int,
    include_title: bool,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped_empty = 0

    with output_path.open("w", encoding="utf-8") as f:
        for example in rows:
            title = str(example.get("title") or "").strip()
            text = str(example.get("text") or "").strip()
            target = str(example.get("summary") or "").strip()

            if not text or not target:
                skipped_empty += 1
                continue

            source_parts = []
            if include_title and title:
                source_parts.append(f"Tiêu đề: {title}")
            source_parts.append(text)

            row = {
                "id": str(example.get("id") or f"{split_name}_{kept:06d}"),
                "source": "\n\n".join(source_parts),
                "target": target,
                "task": task_name,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if max_samples > 0 and kept >= max_samples:
                break

    return kept, skipped_empty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--output_dir", type=Path, default=Path("llm2seq/data/processed/lrsum"))
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--task", default="summarization")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--include_title", action="store_true")
    args = parser.parse_args()

    rows_by_split = load_rows(args.dataset_name, args.language, args.cache_dir)
    for split_name in SPLITS:
        kept, skipped_empty = convert_split(
            rows=rows_by_split[split_name],
            output_path=args.output_dir / f"{split_name}.jsonl",
            split_name=split_name,
            task_name=args.task,
            max_samples=args.max_samples,
            include_title=args.include_title,
        )
        print(
            f"{split_name}: {kept} examples -> {args.output_dir / f'{split_name}.jsonl'} (skipped_empty={skipped_empty})"
        )


if __name__ == "__main__":
    main()
