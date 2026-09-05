from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .schema import _as_text

_SPLITS = {"train": "train", "validation": "validation", "test": "test"}
_LABEL_FILES = {"train": "train.label.jsonl", "validation": "val.label.jsonl", "test": "test.label.jsonl"}
_GENERIC_FILES = {
    "train": ("train.jsonl", "train.json", "train.txt"),
    "validation": ("val.jsonl", "validation.jsonl", "val.json", "validation.json", "val.txt", "validation.txt"),
    "test": ("test.jsonl", "test.json", "test.txt"),
}
_SOURCE_KEYS = ("text", "source", "document", "article_text", "article", "article_text")
_TARGET_KEYS = ("summary", "target", "abstract_text", "abstract", "highlights")


def _first(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if row.get(key) not in (None, "", []):
            return row[key]
    return None


def _iter_json_records(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
            values = value if isinstance(value, list) else [value]
            for row in values:
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must contain objects")
                yield row


def _find_split(input_dir: Path, dataset: str, split: str) -> Path:
    candidates = (_LABEL_FILES[split],) if dataset in {"pubmed", "arxiv"} else _GENERIC_FILES[split]
    for name in candidates:
        path = input_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {split} data in {input_dir}; tried {', '.join(candidates)}")


def _copy_raw(source: Path, raw_dir: Path, split: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / source.name
    if source.resolve() == destination.resolve():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return destination


def _convert(source: Path, destination: Path, split: str, dataset: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    ids: set[str] = set()
    kept = skipped = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for index, row in enumerate(_iter_json_records(source)):
                source_value = _first(row, _SOURCE_KEYS)
                target_value = _first(row, _TARGET_KEYS)
                if source_value is None or target_value is None:
                    skipped += 1
                    continue
                text = _as_text(source_value, "\n").strip()
                summary = _as_text(target_value, "\n").strip()
                if not text or not summary:
                    skipped += 1
                    continue
                example_id = str(row.get("id") or row.get("doc_id") or row.get("article_id") or f"{split}_{index:06d}")
                if example_id in ids:
                    raise ValueError(f"Duplicate id {example_id!r} in {source}")
                ids.add(example_id)
                output.write(
                    json.dumps(
                        {
                            "id": example_id,
                            "text": text,
                            "summary": summary,
                            "task": "summarization",
                            "dataset": dataset,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
        if not kept:
            raise ValueError(f"No valid text/summary pairs found in {source}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"kept": kept, "skipped": skipped, "ids": ids}


def _register_sources(path: Path, split: str, connection: sqlite3.Connection) -> int:
    collisions = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                text = str(json.loads(raw)["text"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid prepared row at {path}:{line_number}") from exc
            existing = connection.execute("SELECT split FROM source_registry WHERE text = ?", (text,)).fetchone()
            if existing is not None:
                if existing[0] != split:
                    collisions += 1
                continue
            connection.execute("INSERT INTO source_registry(text, split) VALUES (?, ?)", (text, split))
    connection.commit()
    return collisions


def prepare_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    raw_copy_dir: str | Path | None = None,
    allow_cross_split_content: bool = False,
) -> dict[str, Any]:
    input_path = Path(input_dir).expanduser().resolve()
    processed = Path(output_dir).expanduser().resolve()
    raw_dir = Path(raw_copy_dir or processed.parent / "raw" / dataset).expanduser().resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(input_path)
    processed.mkdir(parents=True, exist_ok=True)
    registry = processed / ".cross_split_sources.sqlite3"
    registry.unlink(missing_ok=True)
    connection = sqlite3.connect(registry)
    report: dict[str, Any] = {"dataset": dataset, "input_dir": str(input_path), "splits": {}}
    seen_ids: set[str] = set()
    try:
        connection.execute("CREATE TABLE source_registry (text TEXT PRIMARY KEY, split TEXT NOT NULL)")
        for split in ("train", "validation", "test"):
            source = _find_split(input_path, dataset, split)
            copied = _copy_raw(source, raw_dir, split)
            destination = processed / ("validation.jsonl" if split == "validation" else f"{split}.jsonl")
            stats = _convert(copied, destination, split, dataset)
            duplicate_ids = sorted(stats["ids"] & seen_ids)
            duplicate_sources = _register_sources(destination, split, connection)
            if (duplicate_ids or duplicate_sources) and not allow_cross_split_content:
                details = []
                if duplicate_ids:
                    details.append(f"duplicate_ids={duplicate_ids[:5]}")
                if duplicate_sources:
                    details.append(f"duplicate_sources={duplicate_sources}")
                raise ValueError("Cross-split content leakage: " + ", ".join(details))
            seen_ids.update(stats["ids"])
            stats.pop("ids")
            report["splits"][split] = {
                **stats,
                "source_path": str(source),
                "raw_copy": str(copied),
                "processed_path": str(destination),
                "duplicate_ids": len(duplicate_ids),
                "duplicate_sources": duplicate_sources,
            }
        report_path = processed / "preparation_report.json"
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(report_path)
        return report
    finally:
        connection.close()
        registry.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PubMed/ArXiv/CNNDM/WikiLingua JSONL for EviSeq AFMR")
    parser.add_argument("--dataset", required=True, choices=("pubmed", "arxiv", "cnndm", "wikilingua", "custom"))
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-copy-dir", default=None)
    parser.add_argument("--allow-cross-split-content", action="store_true")
    args = parser.parse_args()
    report = prepare_dataset(
        args.input_dir,
        args.output_dir,
        dataset=args.dataset,
        raw_copy_dir=args.raw_copy_dir,
        allow_cross_split_content=args.allow_cross_split_content,
    )
    for split, stats in report["splits"].items():
        print(f"{split}: {stats['kept']} examples (skipped {stats['skipped']}) -> {stats['processed_path']}")


if __name__ == "__main__":
    main()
