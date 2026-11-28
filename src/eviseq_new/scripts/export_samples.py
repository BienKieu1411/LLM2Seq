from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "datasets/pubmed/test.jsonl"


def normalize(text: Any) -> str:
    return " ".join(str(text).split()).lower()


def joined(value: Any) -> str:
    return "\n".join(value) if isinstance(value, list) else str(value)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_documents(path: Path) -> list[dict[str, str]]:
    documents = []
    for index, row in enumerate(read_rows(path)):
        source = row.get("text", row.get("source"))
        reference = row.get("summary", row.get("target", row.get("reference")))
        if source is None or reference is None:
            raise SystemExit(f"{path}:{index + 1} has no source text or no reference summary")
        identifier = str(row.get("id") or row.get("article_id") or f"row-{index + 1:08d}")
        documents.append({"id": identifier, "source": joined(source), "reference": joined(reference)})
    return documents


def by_reference(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    repeated = set()
    for row in rows:
        key = normalize(joined(row["reference"]))
        if key in found:
            repeated.add(key)
        found[key] = row
    for key in repeated:
        del found[key]
    return found


def matched_documents(source: Path, systems: dict[str, Path]) -> list[tuple[dict[str, str], dict[str, str]]]:
    predictions = {label: by_reference(read_rows(path)) for label, path in systems.items()}
    documents = source_documents(source)
    unique = by_reference(documents)
    for label, rows in predictions.items():
        print(f"{label}: {len(rows)} predictions, {len(set(rows) & set(unique))} match a test document")
    matched = []
    for document in documents:
        key = normalize(document["reference"])
        if key in unique and all(key in rows for rows in predictions.values()):
            matched.append((document, {label: rows[key]["prediction"] for label, rows in predictions.items()}))
    return matched


def find_prediction_files(root: Path) -> None:
    for folder, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            if name.endswith(".jsonl") and "prediction" in name.lower():
                path = Path(folder) / name
                with path.open("r", encoding="utf-8") as handle:
                    count = sum(1 for line in handle if line.strip())
                print(f"{count} rows  {path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Save the same test documents and every system's summary to one text file.")
    parser.add_argument("--find", type=Path, help="list the prediction files under this folder and stop")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="test JSONL with text and summary")
    parser.add_argument("--system", action="append", default=[], metavar="LABEL=PREDICTIONS.jsonl")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.find:
        find_prediction_files(args.find)
        return
    if not args.system or not args.output:
        parser.error("give --system LABEL=PATH at least once and --output")
    systems = {}
    for value in args.system:
        label, separator, path = value.partition("=")
        if not separator or not label or not Path(path).is_file():
            raise SystemExit(f"--system expects LABEL=PATH of an existing file, got {value!r}")
        systems[label] = Path(path)
    matched = matched_documents(args.source, systems)
    if len(matched) < args.count:
        raise SystemExit(f"only {len(matched)} documents have a summary from every system; {args.count} needed")
    chosen = sorted(random.Random(args.seed).sample(range(len(matched)), args.count))
    blocks = []
    for number, index in enumerate(chosen, start=1):
        document, summaries = matched[index]
        parts = [f"===== {number}/{args.count} | id: {document['id']} =====", "[SOURCE]", document["source"], "",
                 "[REFERENCE]", document["reference"], ""]
        for label, summary in summaries.items():
            parts += [f"[{label}]", summary, ""]
        blocks.append("\n".join(parts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print(f"{args.count} of {len(matched)} shared documents (seed {args.seed}) -> {args.output}")


if __name__ == "__main__":
    main()
