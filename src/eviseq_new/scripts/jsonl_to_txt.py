#!/usr/bin/env python3
"""Copy JSONL files to .txt without changing their contents.

The .txt files are still JSON Lines internally. This is useful when a service
accepts .txt attachments but rejects the .jsonl extension.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="Input .jsonl files")
    parser.add_argument("-o", "--output-dir", type=Path, help="Destination directory (default: next to each input)")
    args = parser.parse_args()

    pairs: list[tuple[Path, Path]] = []
    for source in args.files:
        if source.suffix.lower() != ".jsonl" or not source.is_file():
            parser.error(f"Not an existing .jsonl file: {source}")
        target_dir = args.output_dir if args.output_dir is not None else source.parent
        target = target_dir / f"{source.stem}.txt"
        if target.exists():
            parser.error(f"Output already exists; refusing to overwrite: {target}")
        pairs.append((source, target))

    if len({target.absolute() for _, target in pairs}) != len(pairs):
        parser.error("Two input files would produce the same .txt output name")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for source, target in pairs:
        created = False
        try:
            with source.open("rb") as original, target.open("xb") as copy:
                created = True
                shutil.copyfileobj(original, copy)
        except Exception:
            if created:
                target.unlink(missing_ok=True)
            raise
        print(f"{source} -> {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
