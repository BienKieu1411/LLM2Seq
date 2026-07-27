#!/usr/bin/env python3
"""Copy and convert local PubMed text/summary JSONL files for T5Gemma."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

from prepare_cnndm_json import iter_records, join_field

SPLIT_FILENAMES: Dict[str, str] = {
    "train": "train.label.jsonl",
    "validation": "val.label.jsonl",
    "test": "test.label.jsonl",
}


def find_split(input_dir: Path, split: str) -> Path:
    candidate = input_dir / SPLIT_FILENAMES[split]
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing {split} split: expected {candidate}")
    return candidate


def copy_raw_file(source: Path, raw_copy_dir: Path) -> Path:
    raw_copy_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_copy_dir / source.name
    if source.resolve() == destination.resolve():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def convert_split(input_path: Path, output_path: Path, split: str) -> tuple[int, int, set[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    identifiers: set[str] = set()
    kept = 0
    skipped = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for index, record in enumerate(iter_records(input_path)):
                source = join_field(record.get("text"), "\n").strip()
                target = join_field(record.get("summary"), "\n").strip()
                if not source or not target:
                    skipped += 1
                    continue
                identifier = str(record.get("id") or record.get("doc_id") or f"{split}_{index:06d}")
                if identifier in identifiers:
                    raise ValueError(f"Duplicate id {identifier!r} in {input_path}")
                identifiers.add(identifier)
                output.write(
                    json.dumps(
                        {
                            "id": identifier,
                            "source": source,
                            "target": target,
                            "task": "summarization",
                            "dataset": "pubmed",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
        if kept == 0:
            raise ValueError(f"No valid text/summary pairs found in {input_path}")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return kept, skipped, identifiers


def prepare(input_dir: Path, raw_copy_dir: Path, output_dir: Path) -> Dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    raw_copy_dir = raw_copy_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    report: Dict[str, Any] = {"dataset": "pubmed", "input_dir": str(input_dir), "splits": {}}
    split_ids: Dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        source_path = find_split(input_dir, split)
        copied_path = copy_raw_file(source_path, raw_copy_dir)
        output_path = output_dir / ("validation.jsonl" if split == "validation" else f"{split}.jsonl")
        kept, skipped, identifiers = convert_split(copied_path, output_path, split)
        split_ids[split] = identifiers
        report["splits"][split] = {
            "kept": kept,
            "skipped": skipped,
            "source_path": str(source_path),
            "raw_copy": str(copied_path),
            "processed_path": str(output_path),
        }

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_ids[left] & split_ids[right]
        if overlap:
            raise ValueError(f"Cross-split ID leakage between {left} and {right}: {sorted(overlap)[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--raw_copy_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    report = prepare(Path(args.input_dir), Path(args.raw_copy_dir), Path(args.output_dir))
    for split, stats in report["splits"].items():
        print(f"{split}: copied {stats['source_path']} -> {stats['raw_copy']}; converted {stats['kept']} examples")


if __name__ == "__main__":
    main()
