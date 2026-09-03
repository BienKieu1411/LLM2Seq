"""Copy and deterministically convert local PubMed summarization files for EviSeq."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator

from .cnndm import iter_records, join_field

SPLIT_FILENAMES: Dict[str, str] = {
    "train": "train.label.jsonl",
    "validation": "val.label.jsonl",
    "test": "test.label.jsonl",
}


def find_split(input_dir: Path, split: str) -> Path:
    filename = SPLIT_FILENAMES[split]
    candidate = input_dir / filename
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing {split} split: expected {candidate}")
    return candidate


def _records_with_unique_ids(path: Path, split: str) -> Iterator[tuple[str, str, str, Any]]:
    identifiers: set[str] = set()
    for index, record in enumerate(iter_records(path)):
        source = join_field(record.get("text"), "\n").strip()
        target = join_field(record.get("summary"), "\n").strip()
        if not source or not target:
            yield "", "", "", None
            continue
        identifier = str(record.get("id") or record.get("doc_id") or record.get("article_id") or f"{split}_{index:06d}")
        if identifier in identifiers:
            raise ValueError(f"Duplicate id {identifier!r} in {path}")
        identifiers.add(identifier)
        label = record.get("label")
        if label is not None and not isinstance(label, list):
            raise ValueError(f"Expected label to be a list of sentence indices in {path}")
        yield identifier, source, target, label


def convert_split(
    input_path: Path,
    output_path: Path,
    split: str,
    *,
    dataset_name: str = "pubmed",
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    identifiers: set[str] = set()
    kept = 0
    skipped = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for identifier, source, target, label in _records_with_unique_ids(input_path, split):
                if not source or not target:
                    skipped += 1
                    continue
                identifiers.add(identifier)
                prepared = {
                    "id": identifier,
                    "source": source,
                    "target": target,
                    "task": "summarization",
                    "dataset": dataset_name,
                }
                if label is not None:
                    prepared["label"] = label
                output.write(json.dumps(prepared, ensure_ascii=False) + "\n")
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


def _register_processed_sources(
    path: Path,
    split: str,
    connection: sqlite3.Connection,
) -> int:
    """Register exact source text on disk and return cross-split collisions.

    This deliberately stores the source text itself instead of a digest.  It
    keeps exact duplicate checking while avoiding a process-sized Python set
    for large PubMed/CNNDM corpora.  A duplicate within the same split is not
    reported here; only a source previously registered under another split is
    a leakage error.
    """

    duplicates = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                source = json.loads(raw)["source"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid processed PubMed record at {path}:{line_number}") from exc
            existing = connection.execute(
                "SELECT split FROM source_registry WHERE source = ?",
                (str(source),),
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != split:
                    duplicates += 1
                continue
            connection.execute(
                "INSERT INTO source_registry(source, split) VALUES (?, ?)",
                (str(source), split),
            )
    connection.commit()
    return duplicates


def prepare(
    input_dir: Path,
    raw_copy_dir: Path,
    output_dir: Path,
    *,
    dataset_name: str = "pubmed",
    allow_cross_split_content: bool = False,
) -> Dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    raw_copy_dir = raw_copy_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"dataset": dataset_name, "input_dir": str(input_dir), "splits": {}}
    seen_identifiers: set[str] = set()
    registry_path = output_dir / ".cross_split_sources.sqlite3"
    registry_path.unlink(missing_ok=True)
    connection = sqlite3.connect(registry_path)
    try:
        connection.execute("CREATE TABLE source_registry (source TEXT PRIMARY KEY, split TEXT NOT NULL)")
        for split in ("train", "validation", "test"):
            source_path = find_split(input_dir, split)
            copied_path = copy_raw_file(source_path, raw_copy_dir)
            output_path = output_dir / ("validation.jsonl" if split == "validation" else f"{split}.jsonl")
            stats = convert_split(copied_path, output_path, split, dataset_name=dataset_name)
            duplicate_ids = sorted(stats["ids"] & seen_identifiers)
            duplicate_source_count = _register_processed_sources(output_path, split, connection)
            if duplicate_ids or duplicate_source_count:
                details = []
                if duplicate_ids:
                    details.append(f"duplicate ids={duplicate_ids[:5]}")
                if duplicate_source_count:
                    details.append(f"duplicate source texts={duplicate_source_count}")
                message = f"Cross-split content leakage while preparing {dataset_name}: " + "; ".join(details)
                if not allow_cross_split_content:
                    raise ValueError(message)
                print(f"WARNING: {message}; continuing because cross-split checks were explicitly disabled")
            seen_identifiers.update(stats["ids"])
            stats.pop("ids")
            report["splits"][split] = {
                **stats,
                "source_path": str(source_path),
                "raw_copy": str(copied_path),
                "processed_path": str(output_path),
            }

        report_path = output_dir / "preparation_report.json"
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(report_path)
        return report
    finally:
        connection.close()
        registry_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--raw-copy-dir", default="data/raw/pubmed")
    parser.add_argument("--output-dir", default="data/pubmed")
    parser.add_argument(
        "--allow-cross-split-content",
        action="store_true",
        default=os.environ.get("EVISEQ_ALLOW_CROSS_SPLIT_CONTENT", "").lower() in {"1", "true", "yes"},
        help="Allow duplicate IDs/source texts across splits (debug only; not valid for paper evaluation)",
    )
    args = parser.parse_args()
    report = prepare(
        Path(args.input_dir),
        Path(args.raw_copy_dir),
        Path(args.output_dir),
        allow_cross_split_content=args.allow_cross_split_content,
    )
    for split, stats in report["splits"].items():
        print(f"{split}: {stats['kept']} examples (skipped {stats['skipped']}) -> {stats['processed_path']}")


if __name__ == "__main__":
    main()
