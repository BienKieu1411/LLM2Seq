#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
from pathlib import Path

def decode_separator(value: str) -> str:
    return value.replace("\\n", "\n").replace("\\t", "\t")

def convert_split(
    input_path: Path,
    output_path: Path,
    split_name: str,
    task_name: str,
    max_samples: int,
    source_joiner: str,
    target_joiner: str,
) -> None:
    text = input_path.read_text(encoding="utf-8-sig")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for line_no, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            source_parts = []
            if "single_documents" in example:
                for doc in example["single_documents"]:
                    doc_parts = []
                    if doc.get("title"):
                        doc_parts.append(f"Title: {doc['title']}")
                    if doc.get("anchor_text"):
                        doc_parts.append(f"Anchor: {doc['anchor_text']}")
                    if doc.get("raw_text"):
                        doc_parts.append(f"Raw: {doc['raw_text']}")
                    source_parts.append("\\n".join(doc_parts))
            elif "text" in example:
                for doc in example.get("text", []):
                    if isinstance(doc, list):
                        source_parts.append(" ".join(doc))
                    elif isinstance(doc, str):
                        source_parts.append(doc)
            source = source_joiner.join(source_parts)
            target_list = example.get("summary", [])
            target = target_joiner.join(target_list) if isinstance(target_list, list) else str(target_list)
            if not source:
                skipped += 1
                continue
            row = {
                "id": f"{split_name}_{kept:06d}",
                "source": source,
                "target": target,
                "task": task_name,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if max_samples > 0 and kept >= max_samples:
                break
    print(f"{split_name}: {kept} examples -> {output_path} (skipped: {skipped})")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="T5Gemma/data/processed/vlsp")
    parser.add_argument("--task", default="summarization")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--source_joiner", default="\\n")
    parser.add_argument("--target_joiner", default=" ")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    source_joiner = decode_separator(args.source_joiner)
    target_joiner = decode_separator(args.target_joiner)
    for filepath in input_dir.glob("*.jsonl"):
        split_name = filepath.name.replace(".jsonl", "")
        if split_name == "vlsp_2022_abmusu":
            continue
        convert_split(
            filepath,
            output_dir / f"{split_name}.jsonl",
            split_name,
            args.task,
            args.max_samples,
            source_joiner,
            target_joiner,
        )

if __name__ == "__main__":
    main()
