#!/usr/bin/env python3
"""Combine SDS/MDS, prepared, and WikiLingua train rows into one dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import unicodedata
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eviseq_afmr.data.schema import CanonicalRecord

ICON = re.compile("[\\U0001f000-\\U0001faff\\u2600-\\u27bf\\ufe0f\\u200d▲▶►◆★☆●■□◉☑✓✔]")


@dataclass(frozen=True)
class TrainSource:
    label: str
    path: Path
    source_field: str
    target_field: str
    id_field: str
    strip_icons: bool = False


def _strip_icons(text: str) -> tuple[str, int]:
    """Remove emoji/icon characters while preserving word boundaries and newlines."""
    normalized = unicodedata.normalize("NFC", text)
    cleaned, removed = ICON.subn(" ", normalized)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned).strip()
    return cleaned, removed


def _record_from_mapping(
    row: dict[str, Any],
    source: TrainSource,
    *,
    icon_counts: dict[str, dict[str, int]] | None = None,
) -> CanonicalRecord:
    record = CanonicalRecord.from_mapping(
        row,
        source_field=source.source_field,
        target_field=source.target_field,
        id_field=source.id_field,
    )
    if not source.strip_icons:
        return record

    clean_source, source_count = _strip_icons(record.source)
    clean_target, target_count = _strip_icons(record.target)
    if icon_counts is not None:
        counts = icon_counts.setdefault(source.label, {"source": 0, "target": 0, "records_with_icons": 0})
        counts["source"] += source_count
        counts["target"] += target_count
        counts["records_with_icons"] += int(source_count > 0 or target_count > 0)
    if not clean_source or not clean_target:
        raise ValueError("source and target must be non-empty")
    return CanonicalRecord(record.example_id, clean_source, clean_target)


def _records(
    source: TrainSource,
    *,
    skip_empty: bool = False,
    empty_counts: dict[str, int] | None = None,
    empty_lines: dict[str, list[int]] | None = None,
    icon_counts: dict[str, dict[str, int]] | None = None,
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
                record = _record_from_mapping(row, source, icon_counts=icon_counts)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                if skip_empty and isinstance(exc, ValueError) and str(exc) == "source and target must be non-empty":
                    if empty_counts is not None:
                        empty_counts[source.label] = empty_counts.get(source.label, 0) + 1
                    if empty_lines is not None and len(empty_lines.setdefault(source.label, [])) < 20:
                        empty_lines[source.label].append(line_number)
                    continue
                raise ValueError(f"Invalid record at {source.path}:{line_number}: {exc}") from exc
            yield offset, line_number, record


def _external_records(source: TrainSource):
    """Read a validation JSON array/object or JSONL file as canonical records."""

    if source.path.suffix.lower() == ".json":
        try:
            with source.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            # Some prepared files use the .json extension for newline-delimited JSON.
            payload = None
        if payload is not None:
            if isinstance(payload, dict):
                for key in ("data", "examples", "records"):
                    if isinstance(payload.get(key), list):
                        payload = payload[key]
                        break
                else:
                    payload = [payload]
            if not isinstance(payload, list):
                raise ValueError(f"Validation JSON must contain an object or list of objects: {source.path}")
            for line_number, row in enumerate(payload, start=1):
                try:
                    if not isinstance(row, dict):
                        raise ValueError("record must be an object")
                    record = CanonicalRecord.from_mapping(
                        row,
                        source_field=source.source_field,
                        target_field=source.target_field,
                        id_field=source.id_field,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid validation record at {source.path}: item {line_number}: {exc}") from exc
                yield line_number, record
            return
    for _, line_number, record in _records(source):
        yield line_number, record


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
    validation_fraction: float | None = None,
    validation_file: Path | None = None,
    validation_source_field: str = "text",
    validation_target_field: str = "summary",
    validation_id_field: str = "id",
) -> dict[str, Any]:
    wiki_dir = wiki_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    wiki_paths = {split: wiki_dir / f"{split}.jsonl" for split in ("train", "validation", "test")}
    if validation_fraction is not None and not 0.0 < validation_fraction < 1.0:
        raise ValueError("WikiLingua validation fraction must be between 0 and 1")
    if validation_fraction is not None and validation_file is not None:
        raise ValueError("Choose either --wikilingua-validation-fraction or --validation-file, not both")
    external_validation = (
        TrainSource(
            "external_validation",
            validation_file.expanduser().resolve(),
            validation_source_field,
            validation_target_field,
            validation_id_field,
        )
        if validation_file is not None
        else None
    )
    required_wiki_splits = (
        ("train", "test")
        if validation_fraction is not None or external_validation is not None
        else ("train", "validation", "test")
    )
    for path in (*(wiki_paths[split] for split in required_wiki_splits), *(source.path for source in extra_sources)):
        if not path.is_file():
            raise FileNotFoundError(path)
    if external_validation is not None and not external_validation.path.is_file():
        raise FileNotFoundError(external_validation.path)
    if (
        output_dir == wiki_dir
        or any(output_dir == source.path.parent for source in extra_sources)
        or (external_validation is not None and output_dir == external_validation.path.parent)
    ):
        raise ValueError("Output directory must differ from every input directory")
    all_sources = [
        *extra_sources,
        TrainSource("wikilingua", wiki_paths["train"], "text", "summary", "id"),
    ]

    validation_entries: list[tuple[int, int]] = []
    validation_source = "wikilingua_validation"
    external_validation_count: int | None = None
    external_validation_source: TrainSource | None = None
    if external_validation is not None:
        test_sources = {
            _source_key(record.source)
            for _, _, record in _records(TrainSource("test", wiki_paths["test"], "text", "summary", "id"))
        }
        validation_sources: set[bytes] = set()
        external_validation_count = 0
        for _, record in _external_records(external_validation):
            validation_sources.add(_source_key(record.source))
            external_validation_count += 1
        if not external_validation_count:
            raise ValueError(f"Validation file is empty: {external_validation.path}")
        overlap = validation_sources & test_sources
        if overlap:
            raise ValueError(f"Validation file shares {len(overlap)} exact source document(s) with WikiLingua test")
        held_out_sources = validation_sources | test_sources
        external_validation_source = external_validation
        validation_source = str(external_validation.path)
    elif validation_fraction is None:
        held_out_sources = {
            _source_key(record.source)
            for split in ("validation", "test")
            for _, _, record in _records(TrainSource(split, wiki_paths[split], "text", "summary", "id"))
        }
    else:
        test_sources = {
            _source_key(record.source)
            for _, _, record in _records(TrainSource("test", wiki_paths["test"], "text", "summary", "id"))
        }
        # Split at the source-document level so duplicate source documents cannot
        # appear in both WikiLingua train and the derived validation set.
        train_groups: dict[bytes, list[tuple[int, int]]] = {}
        for offset, line_number, record in _records(
            TrainSource("wikilingua", wiki_paths["train"], "text", "summary", "id"),
            skip_empty=True,
        ):
            key = _source_key(record.source)
            if key not in test_sources:
                train_groups.setdefault(key, []).append((offset, line_number))
        eligible_count = sum(len(group) for group in train_groups.values())
        if eligible_count < 2 or len(train_groups) < 2:
            raise ValueError(
                "Cannot create WikiLingua validation: need at least two distinct non-test source documents"
            )
        target_count = min(eligible_count - 1, max(1, round(eligible_count * validation_fraction)))
        shuffled_groups = list(train_groups.items())
        random.Random(seed).shuffle(shuffled_groups)
        validation_sources: set[bytes] = set()
        for key, group in shuffled_groups:
            if len(validation_entries) >= target_count:
                break
            if len(validation_entries) + len(group) < eligible_count:
                validation_sources.add(key)
                validation_entries.extend(group)
        if not validation_entries:
            raise ValueError("Could not select WikiLingua validation records while keeping train non-empty")
        held_out_sources = test_sources | validation_sources
        validation_source = "deterministic source-group split from WikiLingua train"
    # Store offsets rather than long documents; shuffled rows are read from their
    # original files only when the canonical output is written.
    entries: list[tuple[int, int, int]] = []
    counts = {source.label: 0 for source in all_sources}
    skipped_overlap = {source.label: 0 for source in all_sources}
    skipped_empty = {source.label: 0 for source in all_sources}
    icon_counts: dict[str, dict[str, int]] = {}
    empty_line_numbers: dict[str, list[int]] = {}
    for source_index, source in enumerate(all_sources):
        for offset, line_number, record in _records(
            source,
            skip_empty=True,
            empty_counts=skipped_empty,
            empty_lines=empty_line_numbers,
            icon_counts=icon_counts,
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
                record = _record_from_mapping(row_data, source)
                row = {
                    "id": f"{source.label}:{line_number}:{record.example_id or line_number}",
                    "text": record.source,
                    "summary": record.target,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_atomic(output_dir / "train.jsonl", write_train)
    if external_validation_source is not None:

        def write_external_validation(temporary: Path) -> None:
            with temporary.open("w", encoding="utf-8") as handle:
                for line_number, record in _external_records(external_validation_source):
                    row = {
                        "id": f"external_validation:{line_number}:{record.example_id or line_number}",
                        "text": record.source,
                        "summary": record.target,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        _write_atomic(output_dir / "validation.jsonl", write_external_validation)
    elif validation_fraction is None:
        _write_atomic(
            output_dir / "validation.jsonl", lambda temporary: shutil.copyfile(wiki_paths["validation"], temporary)
        )
    else:

        def write_validation(temporary: Path) -> None:
            with wiki_paths["train"].open("rb") as input_handle, temporary.open("w", encoding="utf-8") as output_handle:
                for offset, line_number in sorted(validation_entries, key=lambda item: item[1]):
                    input_handle.seek(offset)
                    row_data = json.loads(input_handle.readline())
                    record = CanonicalRecord.from_mapping(
                        row_data,
                        source_field="text",
                        target_field="summary",
                        id_field="id",
                    )
                    row = {
                        "id": f"wikilingua_validation:{line_number}:{record.example_id or line_number}",
                        "text": record.source,
                        "summary": record.target,
                    }
                    output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        _write_atomic(output_dir / "validation.jsonl", write_validation)
    _write_atomic(output_dir / "test.jsonl", lambda temporary: shutil.copyfile(wiki_paths["test"], temporary))
    report = {
        "seed": seed,
        "validation_source": validation_source,
        "validation_fraction_requested": validation_fraction,
        "validation_records": len(validation_entries) if validation_fraction is not None else external_validation_count,
        "validation_fraction_realized": len(validation_entries) / eligible_count
        if validation_fraction is not None
        else None,
        "train_sources": [
            {
                "label": source.label,
                "path": str(source.path),
                "source_field": source.source_field,
                "target_field": source.target_field,
                "records_kept": counts[source.label],
                "records_skipped_empty_source_or_target": skipped_empty[source.label],
                "first_empty_line_numbers": empty_line_numbers.get(source.label, []),
                "records_skipped_for_exact_held_out_source_match": skipped_overlap[source.label],
                "icons_removed": icon_counts.get(source.label, {"source": 0, "target": 0, "records_with_icons": 0}),
            }
            for source in all_sources
        ],
        "train_total": len(entries),
        "validation_split": (
            "external validation file"
            if external_validation_source is not None
            else "byte-identical copy of WikiLingua validation"
            if validation_fraction is None
            else "source-group holdout from WikiLingua train"
        ),
        "test_split": "byte-identical copy of WikiLingua test",
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
    strip_icons: bool = False,
) -> TrainSource:
    return TrainSource(label, path.expanduser().resolve(), source_field, target_field, id_field, strip_icons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-data-dir",
        "--batch-dir",
        dest="summary_data_dir",
        type=Path,
        help="Directory containing sds.jsonl and mds.jsonl (input/abstract_sum fields by default)",
    )
    parser.add_argument("--prepared-train", type=Path, help="Previously prepared train JSONL")
    parser.add_argument(
        "--other-train", type=Path, action="append", default=[], help="Additional train JSONL (can be repeated)"
    )
    parser.add_argument("--wikilingua-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--wikilingua-validation-fraction",
        type=float,
        help="derive validation from this fraction of WikiLingua train, grouped by exact source text",
    )
    parser.add_argument("--validation-file", type=Path, help="use an external validation JSON/JSONL file")
    parser.add_argument("--validation-source-field", default="text")
    parser.add_argument("--validation-target-field", default="summary")
    parser.add_argument("--validation-id-field", default="id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-source-field", "--batch-source-field", dest="batch_source_field", default="input")
    parser.add_argument(
        "--summary-target-field",
        "--batch-target-field",
        dest="batch_target_field",
        default="abstract_sum",
    )
    parser.add_argument("--summary-id-field", "--batch-id-field", dest="batch_id_field", default="id")
    parser.add_argument("--prepared-source-field", default="text")
    parser.add_argument("--prepared-target-field", default="summary")
    parser.add_argument("--prepared-id-field", default="id")
    parser.add_argument("--source-field", default="text", help=argparse.SUPPRESS)
    parser.add_argument("--target-field", default="summary", help=argparse.SUPPRESS)
    parser.add_argument("--id-field", default="id", help=argparse.SUPPRESS)
    args = parser.parse_args()

    extras: list[TrainSource] = []
    if args.summary_data_dir:
        summary_data_dir = args.summary_data_dir.expanduser().resolve()
        for filename in ("sds.jsonl", "mds.jsonl"):
            batch_path = summary_data_dir / filename
            if not batch_path.is_file():
                raise FileNotFoundError(f"Expected training data file: {batch_path}")
            extras.append(
                _source_from_file(
                    batch_path,
                    label=batch_path.stem,
                    source_field=args.batch_source_field,
                    target_field=args.batch_target_field,
                    id_field=args.batch_id_field,
                    strip_icons=True,
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
        parser.error("provide at least one of --summary-data-dir, --prepared-train, or --other-train")

    report = mix(
        args.wikilingua_dir,
        args.output_dir,
        seed=args.seed,
        extra_sources=extras,
        validation_fraction=args.wikilingua_validation_fraction,
        validation_file=args.validation_file,
        validation_source_field=args.validation_source_field,
        validation_target_field=args.validation_target_field,
        validation_id_field=args.validation_id_field,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
