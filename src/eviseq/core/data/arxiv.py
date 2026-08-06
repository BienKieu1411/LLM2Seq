"""Convert ArXiv summarization files for EviSeq.

ArXiv uses the same ``*.label.jsonl`` schema as PubMed: each record contains
``text`` and ``summary`` fields. This module reuses the tested PubMed
converter while recording the correct dataset name in the output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pubmed import prepare as _prepare


def prepare(input_dir: Path, raw_copy_dir: Path, output_dir: Path):
    """Prepare ArXiv files while preserving the shared PubMed schema."""

    return _prepare(input_dir, raw_copy_dir, output_dir, dataset_name="arxiv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--raw-copy-dir", default="data/raw/arxiv")
    parser.add_argument("--output-dir", default="data/arxiv")
    args = parser.parse_args()
    report = prepare(Path(args.input_dir), Path(args.raw_copy_dir), Path(args.output_dir))
    for split, stats in report["splits"].items():
        print(f"{split}: {stats['kept']} examples (skipped {stats['skipped']}) -> {stats['processed_path']}")


if __name__ == "__main__":
    main()
