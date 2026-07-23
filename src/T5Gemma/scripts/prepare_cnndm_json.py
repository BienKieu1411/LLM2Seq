#!/usr/bin/env python3
"""Copy and convert local CNN/DailyMail JSONL files for T5Gemma."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Sequence


SPLIT_CANDIDATES: Dict[str, Sequence[str]] = {
    "train": ("train.txt", "train.jsonl", "train.json"),
    "validation": (
        "val.txt",
        "validation.txt",
        "val.jsonl",
        "validation.jsonl",
        "val.json",
        "validation.json",
    ),
    "test": ("test.txt", "test.jsonl", "test.json"),
}
RAW_FILENAMES = {
    "train": "train.txt",
    "validation": "val.txt",
    "test": "test.txt",
}


def find_split(input_dir: Path, split: str) -> Path:
    for filename in SPLIT_CANDIDATES[split]:
        candidate = input_dir / filename
        if candidate.is_file():
            return candidate
    expected = ", ".join(SPLIT_CANDIDATES[split])
    raise FileNotFoundError(f"Missing {split} split in {input_dir}; expected one of: {expected}")


def iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream JSONL records, while also accepting one-record-per-line arrays."""

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} is not valid one-record-per-line JSON: {exc}"
                ) from exc
            if isinstance(value, dict):
                yield value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
            else:
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object, got {type(value).__name__}"
                )


def _detokenize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+([\)\]\}])", r"\1", text)
    text = re.sub(r"\s+n['’]t\b", "n't", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(['’](?:s|re|ve|ll|d|m))\b", r"\1", text, flags=re.IGNORECASE)
    return text.strip()


def join_field(value: Any, separator: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _detokenize(value)
    if isinstance(value, (list, tuple)):
        parts = [join_field(item, " ") for item in value]
        return separator.join(part for part in parts if part)
    if isinstance(value, dict):
        parts = [join_field(item, " ") for item in value.values()]
        return separator.join(part for part in parts if part)
    return _detokenize(str(value))


def first_present(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "" and value != []:
            return value
    return None


def convert_split(input_path: Path, output_path: Path, split: str) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    kept = 0
    skipped = 0
    with temporary.open("w", encoding="utf-8") as output:
        for index, record in enumerate(iter_records(input_path)):
            article = first_present(
                record,
                ("article_text", "article", "document", "source", "src"),
            )
            abstract = first_present(
                record,
                ("abstract_text", "abstract", "highlights", "summary", "target", "tgt"),
            )
            source = join_field(article, "\n")
            target = join_field(abstract, "\n")
            target = re.sub(r"(?:^|\n)\s*@highlight\s*(?:\n|$)", "\n", target, flags=re.IGNORECASE)
            source = source.strip()
            target = target.strip()
            if not source or not target:
                skipped += 1
                continue
            identifier = (
                record.get("id")
                or record.get("doc_id")
                or record.get("article_id")
                or f"{split}_{index:06d}"
            )
            output.write(
                json.dumps(
                    {
                        "id": str(identifier),
                        "source": source,
                        "target": target,
                        "task": "summarization",
                        "dataset": "cnndm",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
    if kept == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"No valid article/abstract pairs found in {input_path}")
    temporary.replace(output_path)
    return kept, skipped


def copy_raw_file(source: Path, raw_copy_dir: Path, split: str) -> Path:
    raw_copy_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_copy_dir / RAW_FILENAMES[split]
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--raw_copy_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    raw_copy_dir = Path(args.raw_copy_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    for split in ("train", "validation", "test"):
        source_path = find_split(input_dir, split)
        copied_path = copy_raw_file(source_path, raw_copy_dir, split)
        output_name = "validation.jsonl" if split == "validation" else f"{split}.jsonl"
        kept, skipped = convert_split(copied_path, output_dir / output_name, split)
        print(
            f"{split}: copied {source_path} -> {copied_path}; "
            f"converted {kept} examples (skipped {skipped})"
        )


if __name__ == "__main__":
    main()
