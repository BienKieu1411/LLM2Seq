"""Copy and deterministically convert local CNN/DailyMail files for V5."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Sequence

from .data import dataset_fingerprint

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
RAW_FILENAMES = {"train": "train.txt", "validation": "val.txt", "test": "test.txt"}
SOURCE_FIELDS = ("article_text", "article", "document", "source", "src")
TARGET_FIELDS = ("abstract_text", "abstract", "highlights", "summary", "target", "tgt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_split(input_dir: Path, split: str) -> Path:
    for filename in SPLIT_CANDIDATES[split]:
        candidate = input_dir / filename
        if candidate.is_file():
            return candidate
    expected = ", ".join(SPLIT_CANDIDATES[split])
    raise FileNotFoundError(f"Missing {split} split in {input_dir}; expected one of: {expected}")


def iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream JSON objects or one-line arrays from a JSONL-style file."""

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                hint = " Files must contain one JSON record per line; pretty-printed multi-line arrays are unsupported."
                raise ValueError(f"{path}:{line_number} is not valid JSONL: {exc}.{hint}") from exc
            values = value if isinstance(value, list) else [value]
            for item in values:
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{line_number} must contain JSON objects, got {type(item).__name__}")
                yield item


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


def _token_sequence(values: Sequence[Any]) -> bool:
    """Detect token arrays so they do not become one source unit per token."""

    strings = [str(value).strip() for value in values if str(value).strip()]
    if len(strings) < 8:
        return False
    single_token = sum(len(value.split()) == 1 for value in strings) / len(strings)
    sentence_like = sum(bool(re.search(r"[.!?][\"']?$", value)) for value in strings) / len(strings)
    return single_token >= 0.80 and sentence_like < 0.20


def join_field(value: Any, separator: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _detokenize(value)
    if isinstance(value, (list, tuple)):
        child_separator = " " if _token_sequence(value) else separator
        parts = [join_field(item, " ") for item in value]
        return child_separator.join(part for part in parts if part)
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


def convert_split(input_path: Path, output_path: Path, split: str) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    kept = 0
    skipped = 0
    identifiers: set[str] = set()
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for index, record in enumerate(iter_records(input_path)):
                source = join_field(first_present(record, SOURCE_FIELDS), "\n").strip()
                target = join_field(first_present(record, TARGET_FIELDS), "\n")
                target = re.sub(r"(?:^|\n)\s*@highlight\s*(?:\n|$)", "\n", target, flags=re.IGNORECASE)
                target = target.strip()
                if not source or not target:
                    skipped += 1
                    continue
                identifier = str(
                    record.get("id") or record.get("doc_id") or record.get("article_id") or f"{split}_{index:06d}"
                )
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
                            "dataset": "cnndm",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
        if kept == 0:
            raise ValueError(f"No valid article/summary pairs found in {input_path}")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"kept": kept, "skipped": skipped, "ids": identifiers}


def copy_raw_file(source: Path, raw_copy_dir: Path, split: str) -> Path:
    raw_copy_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_copy_dir / RAW_FILENAMES[split]
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

    report: Dict[str, Any] = {"dataset": "cnndm", "input_dir": str(input_dir), "splits": {}}
    split_ids: Dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        source_path = find_split(input_dir, split)
        copied_path = copy_raw_file(source_path, raw_copy_dir, split)
        output_name = "validation.jsonl" if split == "validation" else f"{split}.jsonl"
        output_path = output_dir / output_name
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
            sample = sorted(overlap)[:5]
            raise ValueError(f"Cross-split ID leakage between {left} and {right}: {sample}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--raw-copy-dir", default="data/raw/cnndm")
    parser.add_argument("--output-dir", default="data/cnndm")
    args = parser.parse_args()
    report = prepare(Path(args.input_dir), Path(args.raw_copy_dir), Path(args.output_dir))
    for split, stats in report["splits"].items():
        print(f"{split}: {stats['kept']} examples (skipped {stats['skipped']}) -> {stats['processed_path']}")


if __name__ == "__main__":
    main()
