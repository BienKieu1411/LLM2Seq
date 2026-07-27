"""Copy and deterministically convert the local PubMed summarization files for V4."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

from .data import dataset_fingerprint
from .prepare_cnndm import iter_records, join_field

SPLIT_FILENAMES: Dict[str, str] = {
    "train": "train.label.jsonl",
    "validation": "val.label.jsonl",
    "test": "test.label.jsonl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_split(input_dir: Path, split: str) -> Path:
    filename = SPLIT_FILENAMES[split]
    candidate = input_dir / filename
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing {split} split: expected {candidate}")
    return candidate


def _records_with_unique_ids(path: Path, split: str) -> Iterator[tuple[str, str, str]]:
    identifiers: set[str] = set()
    for index, record in enumerate(iter_records(path)):
        source = join_field(record.get("text"), "\n").strip()
        target = join_field(record.get("summary"), "\n").strip()
        if not source or not target:
            yield "", "", ""
            continue
        identifier = str(record.get("id") or record.get("doc_id") or f"{split}_{index:06d}")
        if identifier in identifiers:
            raise ValueError(f"Duplicate id {identifier!r} in {path}")
        identifiers.add(identifier)
        yield identifier, source, target


def convert_split(input_path: Path, output_path: Path, split: str) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    identifiers: set[str] = set()
    kept = 0
    skipped = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for identifier, source, target in _records_with_unique_ids(input_path, split):
                if not source or not target:
                    skipped += 1
                    continue
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
    return {"kept": kept, "skipped": skipped, "ids": identifiers}


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
        stats = convert_split(copied_path, output_path, split)
        split_ids[split] = stats.pop("ids")
        report["splits"][split] = {
            **stats,
            "source_path": str(source_path),
            "raw_copy": str(copied_path),
            "raw_sha256": _sha256(copied_path),
            "processed_path": str(output_path),
            "processed_file_sha256": _sha256(output_path),
            "canonical_fingerprint": dataset_fingerprint(output_path),
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
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--raw-copy-dir", default="data/raw/pubmed")
    parser.add_argument("--output-dir", default="data/pubmed")
    args = parser.parse_args()
    report = prepare(Path(args.input_dir), Path(args.raw_copy_dir), Path(args.output_dir))
    for split, stats in report["splits"].items():
        print(f"{split}: {stats['kept']} examples (skipped {stats['skipped']}) -> {stats['processed_path']}")


if __name__ == "__main__":
    main()
