#!/usr/bin/env python3
"""Convert local WikiLingua train/val/test JSON files to LLM2Seq JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def decode_separator(value: str) -> str:
    return value.replace("\\n", "\n").replace("\\t", "\t")


def join_text(value: Any, list_sep: str = "\n") -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return list_sep.join(part for part in (join_text(item, list_sep=list_sep) for item in value) if part)
    if isinstance(value, dict):
        return list_sep.join(part for part in (join_text(item, list_sep=list_sep) for item in value.values()) if part)
    return str(value).strip()


def normalize_records(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "src" in raw and "tgt" in raw:
            return [raw]
        for key in ("data", "examples", "records"):
            if isinstance(raw.get(key), list):
                return raw[key]
        values = list(raw.values())
        if values and all(isinstance(value, dict) for value in values):
            return values
    raise ValueError("Expected list[dict], dict with src/tgt, or dict of records")


def load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    decoder = json.JSONDecoder()
    values = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        value, idx = decoder.raw_decode(text, idx)
        values.append(value)

    if len(values) == 1:
        return normalize_records(values[0])

    records: List[Dict[str, Any]] = []
    for value in values:
        records.extend(normalize_records(value))
    return records


def convert_split(
    input_path: Path,
    output_path: Path,
    split_name: str,
    task_name: str,
    max_samples: int,
    source_joiner: str,
    target_joiner: str,
) -> None:
    records = load_records(input_path)
    if max_samples > 0:
        records = records[:max_samples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, example in enumerate(records):
            source = join_text(example.get("src") or example.get("source"), list_sep=source_joiner)
            target = join_text(
                example.get("tgt") or example.get("target") or example.get("summary"),
                list_sep=target_joiner,
            )
            if not source or not target:
                skipped += 1
                continue
            row = {
                "id": f"{split_name}_{idx:06d}",
                "source": source,
                "target": target,
                "task": task_name,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"{split_name}: {kept}/{len(records)} examples -> {output_path} (skipped: {skipped})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder containing train.json, val.json, test.json")
    parser.add_argument("--output_dir", default="llm2seq_h200/data/processed")
    parser.add_argument("--task", default="summarization")
    parser.add_argument("--max_train", type=int, default=-1)
    parser.add_argument("--max_eval", type=int, default=-1)
    parser.add_argument("--source_joiner", default="\\n")
    parser.add_argument("--target_joiner", default=" ")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    train_file = input_dir / "train.json"
    val_file = input_dir / "val.json"
    if not val_file.exists():
        val_file = input_dir / "validation.json"
    test_file = input_dir / "test.json"

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError("input_dir must contain train.json and val.json/validation.json")

    source_joiner = decode_separator(args.source_joiner)
    target_joiner = decode_separator(args.target_joiner)

    convert_split(
        train_file,
        output_dir / "train.jsonl",
        "train",
        args.task,
        args.max_train,
        source_joiner,
        target_joiner,
    )
    convert_split(
        val_file,
        output_dir / "validation.jsonl",
        "validation",
        args.task,
        args.max_eval,
        source_joiner,
        target_joiner,
    )
    if test_file.exists():
        convert_split(
            test_file,
            output_dir / "test.jsonl",
            "test",
            args.task,
            args.max_eval,
            source_joiner,
            target_joiner,
        )


if __name__ == "__main__":
    main()
