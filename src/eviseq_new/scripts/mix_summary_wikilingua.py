#!/usr/bin/env python3
"""Create one shuffled training split; keep WikiLingua validation/test unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

from eviseq_afmr.data.schema import CanonicalRecord


def _records(path: Path, *, source_field: str, target_field: str, id_field: str):
    with path.open("rb") as handle:
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
                    row, source_field=source_field, target_field=target_field, id_field=id_field
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
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
    other_train: Path,
    wiki_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    source_field: str,
    target_field: str,
    id_field: str,
) -> dict:
    other_train = other_train.expanduser().resolve()
    wiki_dir = wiki_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    wiki_paths = {split: wiki_dir / f"{split}.jsonl" for split in ("train", "validation", "test")}
    for path in (other_train, *wiki_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    if other_train == wiki_paths["train"]:
        raise ValueError("The other training file must differ from WikiLingua train")
    if output_dir in {wiki_dir, other_train.parent}:
        raise ValueError("Output directory must differ from both input directories")

    held_out_sources = {
        _source_key(record.source)
        for split in ("validation", "test")
        for _, _, record in _records(wiki_paths[split], source_field="text", target_field="summary", id_field="id")
    }
    # Keep only file offsets in memory; long sources are read again while writing.
    entries: list[tuple[str, int, int]] = []
    counts = {"other": 0, "wiki": 0}
    skipped_overlap = {"other": 0, "wiki": 0}
    for origin, path, fields in (
        ("other", other_train, (source_field, target_field, id_field)),
        ("wiki", wiki_paths["train"], ("text", "summary", "id")),
    ):
        for offset, line_number, record in _records(
            path, source_field=fields[0], target_field=fields[1], id_field=fields[2]
        ):
            if _source_key(record.source) in held_out_sources:
                skipped_overlap[origin] += 1
                continue
            entries.append((origin, offset, line_number))
            counts[origin] += 1
    if not counts["other"] or not counts["wiki"]:
        raise ValueError(f"Both train sources must have usable rows; counts={counts}")
    random.Random(seed).shuffle(entries)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_train(temporary: Path) -> None:
        with (
            other_train.open("rb") as other_handle,
            wiki_paths["train"].open("rb") as wiki_handle,
            temporary.open("w", encoding="utf-8") as handle,
        ):
            for origin, offset, line_number in entries:
                source_handle = other_handle if origin == "other" else wiki_handle
                source_handle.seek(offset)
                row_data = json.loads(source_handle.readline())
                fields = (source_field, target_field, id_field) if origin == "other" else ("text", "summary", "id")
                record = CanonicalRecord.from_mapping(
                    row_data, source_field=fields[0], target_field=fields[1], id_field=fields[2]
                )
                row = {
                    "id": f"{origin}:{line_number}:{record.example_id or line_number}",
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
        "other_train": str(other_train),
        "wiki_dir": str(wiki_dir),
        "seed": seed,
        "train_counts": counts,
        "skipped_train_sources_matching_wiki_validation_or_test": skipped_overlap,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--other-train", type=Path, required=True)
    parser.add_argument("--wikilingua-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-field", default="text")
    parser.add_argument("--target-field", default="summary")
    parser.add_argument("--id-field", default="id")
    args = parser.parse_args()
    report = mix(
        args.other_train,
        args.wikilingua_dir,
        args.output_dir,
        seed=args.seed,
        source_field=args.source_field,
        target_field=args.target_field,
        id_field=args.id_field,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
