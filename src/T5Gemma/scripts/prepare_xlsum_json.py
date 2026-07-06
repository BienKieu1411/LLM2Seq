#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Iterable


SPLITS = {
    "vietnamese_train.jsonl": "train",
    "vietnamese_val.jsonl": "validation",
    "vietnamese_test.jsonl": "test",
}


def token_count(text: str) -> int:
    return len(text.strip().split())


def iter_archive_lines(archive_path: Path, member_name: str) -> Iterable[str]:
    with tarfile.open(archive_path, "r:*") as tar:
        member = next((m for m in tar.getmembers() if Path(m.name).name == member_name), None)
        if member is None:
            raise FileNotFoundError(f"{member_name} not found in {archive_path}")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Cannot read {member.name} from {archive_path}")
        for raw in extracted:
            yield raw.decode("utf-8-sig").strip()


def iter_input_lines(input_dir: Path, filename: str) -> Iterable[str]:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    yield from path.read_text(encoding="utf-8-sig").splitlines()


def convert_split(
    lines: Iterable[str],
    output_path: Path,
    split_name: str,
    task_name: str,
    min_target_tokens: int,
    max_samples: int,
    include_title: bool,
) -> tuple[int, int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped_empty = 0
    skipped_short_target = 0

    with output_path.open("w", encoding="utf-8") as f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            title = str(example.get("title") or "").strip()
            text = str(example.get("text") or "").strip()
            target = str(example.get("summary") or "").strip()

            if not text or not target:
                skipped_empty += 1
                continue
            if min_target_tokens > 0 and token_count(target) < min_target_tokens:
                skipped_short_target += 1
                continue

            source_parts = []
            if include_title and title:
                source_parts.append(f"Tiêu đề: {title}")
            source_parts.append(text)

            row = {
                "id": str(example.get("id") or f"{split_name}_{kept:06d}"),
                "source": "\n\n".join(source_parts),
                "target": target,
                "task": task_name,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if max_samples > 0 and kept >= max_samples:
                break

    return kept, skipped_empty, skipped_short_target


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("T5Gemma/data/processed/xlsum"))
    parser.add_argument("--task", default="summarization")
    parser.add_argument("--min_target_tokens", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--no_title", action="store_true")
    args = parser.parse_args()

    for filename, split_name in SPLITS.items():
        if args.archive is not None:
            lines = iter_archive_lines(args.archive, filename)
        else:
            lines = iter_input_lines(args.input_dir, filename)
        kept, skipped_empty, skipped_short_target = convert_split(
            lines=lines,
            output_path=args.output_dir / f"{split_name}.jsonl",
            split_name=split_name,
            task_name=args.task,
            min_target_tokens=args.min_target_tokens,
            max_samples=args.max_samples,
            include_title=not args.no_title,
        )
        print(
            f"{split_name}: {kept} examples -> {args.output_dir / f'{split_name}.jsonl'} "
            f"(skipped_empty={skipped_empty}, skipped_short_target={skipped_short_target})"
        )


if __name__ == "__main__":
    main()
