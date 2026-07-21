#!/usr/bin/env python3
"""Convert local WikiLingua train/validation/test JSON files to GenBridge JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from genbridge.data import clean_wikihow_metadata


def _join_text(value: Any, separator: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return separator.join(
            part for part in (_join_text(item, separator) for item in value) if part
        )
    if isinstance(value, dict):
        return separator.join(
            part for part in (_join_text(item, separator) for item in value.values()) if part
        )
    return str(value).strip()


def _normalize_records(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if any(key in raw for key in ("src", "source")):
            return [raw]
        for key in ("data", "examples", "records"):
            if isinstance(raw.get(key), list):
                return raw[key]
        values = list(raw.values())
        if values and all(isinstance(value, dict) for value in values):
            return values
    raise ValueError("Expected a record list, a data/examples/records list, or a record mapping")


def _load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    decoder = json.JSONDecoder()
    values = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        value, position = decoder.raw_decode(text, position)
        values.append(value)
    if len(values) == 1:
        return _normalize_records(values[0])
    records: List[Dict[str, Any]] = []
    for value in values:
        records.extend(_normalize_records(value))
    return records


def _convert(
    input_path: Path,
    output_path: Path,
    split: str,
    clean_metadata: bool,
) -> None:
    records = _load_records(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            source = _join_text(record.get("src") or record.get("source"), "\n")
            target = _join_text(
                record.get("tgt") or record.get("target") or record.get("summary"),
                " ",
            )
            if clean_metadata:
                # WikiLingua-VI contains serialized WikiHow image/licensing
                # objects inside both documents and references. They are page
                # metadata, not summary content.
                source = clean_wikihow_metadata(source)
                target = clean_wikihow_metadata(target)
            if not source or not target:
                continue
            handle.write(
                json.dumps(
                    {
                        "id": str(record.get("id") or f"{split}_{index:06d}"),
                        "source": source,
                        "target": target,
                        "task": "summarization",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
    print(f"{split}: {kept}/{len(records)} examples -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/wikilingua"))
    parser.add_argument(
        "--keep-wikihow-metadata",
        action="store_true",
        help="Keep embedded smallUrl/licensing blobs (not recommended)",
    )
    args = parser.parse_args()
    validation = args.input_dir / "val.json"
    if not validation.exists():
        validation = args.input_dir / "validation.json"
    files = {
        "train": args.input_dir / "train.json",
        "validation": validation,
        "test": args.input_dir / "test.json",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing WikiLingua files: " + ", ".join(missing))
    for split, path in files.items():
        _convert(
            path,
            args.output_dir / f"{split}.jsonl",
            split,
            clean_metadata=not args.keep_wikihow_metadata,
        )


if __name__ == "__main__":
    main()
