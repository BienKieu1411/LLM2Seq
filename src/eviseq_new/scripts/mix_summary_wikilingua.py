#!/usr/bin/env python3
"""Combine batch, prepared, and WikiLingua train rows into one canonical dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eviseq_afmr.data.schema import CanonicalRecord


@dataclass(frozen=True)
class TrainSource:
    label: str
    path: Path
    source_field: str
    target_field: str
    id_field: str


def _records(
    source: TrainSource,
    *,
    skip_empty: bool = False,
    empty_counts: dict[str, int] | None = None,
    empty_lines: dict[str, list[int]] | None = None,
):
    with source.path.open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("record must be an object")
                record = CanonicalRecord.from_mapping(
                    row,
                    source_field=source.source_field,
                    target_field=source.target_field,
                    id_field=source.id_field,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                if skip_empty and isinstance(exc, ValueError) and str(exc) == "source and target must be non-empty":
                    if empty_counts is not None:
                        empty_counts[source.label] = empty_counts.get(source.label, 0) + 1
                    if empty_lines is not None and len(empty_lines.setdefault(source.label, [])) < 20:
                        empty_lines[source.label].append(line_number)
                    continue
                raise ValueError(f"Invalid record at {source.path}:{line_number}: {exc}") from exc
            yield offset, line_number, record


def _source_key(source: str) -> bytes:
    return hashlib.sha256(source.strip().encode("utf-8")).digest()


def _write_atomic(destination: Path, write) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        write(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def mix(
    wiki_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    extra_sources: list[TrainSource],
) -> dict[str, Any]:
    wiki_dir = wiki_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    wiki_paths = {split: wiki_dir / f"{split}.jsonl" for split in ("train", "validation", "test")}
    for path in (*wiki_paths.values(), *(source.path for source in extra_sources)):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir == wiki_dir or any(output_dir == source.path.parent for source in extra_sources):
        raise ValueError("Output directory must differ from every input directory")
    all_sources = [
        *extra_sources,
        TrainSource("wikilingua", wiki_paths["train"], "text", "summary", "id"),
    ]

    held_out_sources = {
        _source_key(record.source)
        for split in ("validation", "test")
        for _, _, record in _records(TrainSource(split, wiki_paths[split], "text", "summary", "id"))
    }
    # Store offsets rather than long documents; shuffled rows are read from their
    # original files only when the canonical output is written.
    entries: list[tuple[int, int, int]] = []
    counts = {source.label: 0 for source in all_sources}
    skipped_overlap = {source.label: 0 for source in all_sources}
    skipped_empty = {source.label: 0 for source in all_sources}
    empty_line_numbers: dict[str, list[int]] = {}
    for source_index, source in enumerate(all_sources):
        for offset, line_number, record in _records(
            source,
            skip_empty=True,
            empty_counts=skipped_empty,
            empty_lines=empty_line_numbers,
        ):
            if _source_key(record.source) in held_out_sources:
                skipped_overlap[source.label] += 1
                continue
            entries.append((source_index, offset, line_number))
            counts[source.label] += 1
    if not any(counts[source.label] for source in extra_sources):
        raise ValueError("No usable records remain in the additional training sources")
    if not counts["wikilingua"]:
        raise ValueError("No usable WikiLingua training records remain")

    random.Random(seed).shuffle(entries)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_train(temporary: Path) -> None:
        with ExitStack() as stack, temporary.open("w", encoding="utf-8") as handle:
            input_handles = [stack.enter_context(source.path.open("rb")) for source in all_sources]
            for source_index, offset, line_number in entries:
                source = all_sources[source_index]
                input_handle = input_handles[source_index]
                input_handle.seek(offset)
                row_data = json.loads(input_handle.readline())
                record = CanonicalRecord.from_mapping(
                    row_data,
                    source_field=source.source_field,
                    target_field=source.target_field,
                    id_field=source.id_field,
                )
                row = {
                    "id": f"{source.label}:{line_number}:{record.example_id or line_number}",
                    "text": record.source,
                    "summary": record.target,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_atomic(output_dir / "train.jsonl", write_train)
    for split in ("validation", "test"):
        _write_atomic(
            output_dir / f"{split}.jsonl", lambda temporary, split=split: shutil.copyfile(wiki_paths[split], temporary)
        )
    report = {
        "seed": seed,
        "train_sources": [
            {
                "label": source.label,
                "path": str(source.path),
                "source_field": source.source_field,
                "target_field": source.target_field,
                "records_kept": counts[source.label],
                "records_skipped_empty_source_or_target": skipped_empty[source.label],
                "first_empty_line_numbers": empty_line_numbers.get(source.label, []),
                "records_skipped_for_exact_wiki_validation_or_test_source_match": skipped_overlap[source.label],
            }
            for source in all_sources
        ],
        "train_total": len(entries),
        "validation_test": "byte-identical copies of WikiLingua splits",
    }
    _write_atomic(
        output_dir / "mix_report.json",
        lambda temporary: temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        ),
    )
    return report


def _source_from_file(
    path: Path,
    *,
    label: str,
    source_field: str,
    target_field: str,
    id_field: str,
) -> TrainSource:
    return TrainSource(label, path.expanduser().resolve(), source_field, target_field, id_field)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, help="Directory containing batch*.jsonl with input/output fields")
    parser.add_argument("--batch-pattern", default="batch*.jsonl")
    parser.add_argument("--prepared-train", type=Path, help="Previously prepared train JSONL")
    parser.add_argument(
        "--other-train", type=Path, action="append", default=[], help="Additional train JSONL (can be repeated)"
    )
    parser.add_argument("--wikilingua-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-source-field", default="input")
    parser.add_argument("--batch-target-field", default="output")
    parser.add_argument("--batch-id-field", default="id")
    parser.add_argument("--prepared-source-field", default="text")
    parser.add_argument("--prepared-target-field", default="summary")
    parser.add_argument("--prepared-id-field", default="id")
    parser.add_argument("--source-field", default="text", help=argparse.SUPPRESS)
    parser.add_argument("--target-field", default="summary", help=argparse.SUPPRESS)
    parser.add_argument("--id-field", default="id", help=argparse.SUPPRESS)
    args = parser.parse_args()

    extras: list[TrainSource] = []
    if args.batch_dir:
        batch_dir = args.batch_dir.expanduser().resolve()
        batch_files = sorted(batch_dir.glob(args.batch_pattern))
        if not batch_files:
            raise FileNotFoundError(f"No files matching {args.batch_pattern!r} in {batch_dir}")
        for batch_path in batch_files:
            extras.append(
                _source_from_file(
                    batch_path,
                    label=f"batch:{batch_path.stem}",
                    source_field=args.batch_source_field,
                    target_field=args.batch_target_field,
                    id_field=args.batch_id_field,
                )
            )
    if args.prepared_train:
        extras.append(
            _source_from_file(
                args.prepared_train,
                label="prepared_train",
                source_field=args.prepared_source_field,
                target_field=args.prepared_target_field,
                id_field=args.prepared_id_field,
            )
        )
    for index, path in enumerate(args.other_train):
        extras.append(
            _source_from_file(
                path,
                label=f"other_train_{index + 1}",
                source_field=args.source_field,
                target_field=args.target_field,
                id_field=args.id_field,
            )
        )
    if not extras:
        parser.error("provide at least one of --batch-dir, --prepared-train, or --other-train")

    report = mix(args.wikilingua_dir, args.output_dir, seed=args.seed, extra_sources=extras)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
