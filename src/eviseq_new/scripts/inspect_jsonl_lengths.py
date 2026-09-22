"""Inspect sample counts and source/target token-length distributions in JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from jsonl_length_utils import (
    compact_value,
    flatten_fields,
    iter_jsonl,
    load_counters,
    record_texts,
    summarize_lengths,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL file to inspect")
    parser.add_argument("--tokenizer", help="One local tokenizer used for both source and target")
    parser.add_argument("--source-tokenizer", help="Tokenizer used for source lengths")
    parser.add_argument("--target-tokenizer", help="Tokenizer used for target lengths")
    parser.add_argument("--source-field", default="text")
    parser.add_argument("--target-field", default="summary")
    parser.add_argument("--list-separator", default="\n")
    parser.add_argument(
        "--max-target-tokens", type=int, default=512, help="Report the number of targets <= this length"
    )
    parser.add_argument("--output-dir", type=Path, help="Write JSON report, CSV lengths, and a PNG histogram here")
    parser.add_argument("--plot", type=Path, help="Optional explicit PNG path for the histogram")
    parser.add_argument("--limit", type=int, default=0, help="Inspect only the first N valid records (0 means all)")
    parser.add_argument("--strict", action="store_true", help="Fail on malformed rows or missing text fields")
    parser.add_argument(
        "--sample-rows", type=int, default=3, help="Show this many complete sample objects (0 disables samples)"
    )
    parser.add_argument(
        "--sample-value-chars", type=int, default=500, help="Truncate long sample values; <=0 keeps complete values"
    )
    parser.add_argument("--no-schema", action="store_true", help="Do not print field names/types in the report")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face tokenizer downloads")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _plot_lengths(source: list[int], target: list[int], destination: Path, target_limit: int | None) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib is not installed; skipped histogram", file=sys.stderr)
        return False

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(source, bins=50, color="#4c78a8", alpha=0.85)
    axes[0].set_title("Source length")
    axes[1].hist(target, bins=50, color="#f58518", alpha=0.85)
    if target_limit is not None:
        axes[1].axvline(target_limit, color="#d62728", linestyle="--", label=f"limit={target_limit}")
        axes[1].legend()
    axes[1].set_title("Target length")
    for axis in axes:
        axis.set_xlabel("tokens")
        axis.set_ylabel("samples")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return True


def inspect(args: argparse.Namespace) -> dict:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    source_counter, target_counter = load_counters(
        tokenizer=args.tokenizer,
        source_tokenizer=args.source_tokenizer,
        target_tokenizer=args.target_tokenizer,
        allow_download=args.allow_download,
        trust_remote_code=args.trust_remote_code,
    )
    source_lengths: list[int] = []
    target_lengths: list[int] = []
    rows: list[dict[str, int | str]] = []
    sample_records: list[dict] = []
    field_counts: dict[str, int] = {}
    field_types: dict[str, set[str]] = {}
    invalid = 0
    for line_number, _raw_line, row in iter_jsonl(args.input):
        for field, value in flatten_fields(row).items():
            field_counts[field] = field_counts.get(field, 0) + 1
            field_types.setdefault(field, set()).add(type(value).__name__)
        if args.sample_rows > len(sample_records):
            sample_records.append(
                {field: compact_value(value, args.sample_value_chars) for field, value in row.items()}
            )
        try:
            source, target = record_texts(
                row,
                source_field=args.source_field,
                target_field=args.target_field,
                separator=args.list_separator,
            )
            if not source or not target:
                raise ValueError("source and target must be non-empty")
            source_length = source_counter(source)
            target_length = target_counter(target)
        except (TypeError, ValueError, KeyError) as exc:
            invalid += 1
            if args.strict:
                raise ValueError(f"{args.input}:{line_number}: {exc}") from exc
            continue
        source_lengths.append(source_length)
        target_lengths.append(target_length)
        rows.append(
            {
                "line": line_number,
                "id": str(row.get("id", line_number)),
                "source_tokens": source_length,
                "target_tokens": target_length,
            }
        )
        if args.limit and len(rows) >= args.limit:
            break

    report = {
        "input": str(args.input.resolve()),
        "samples": len(rows),
        "invalid_rows": invalid,
        "source_tokenizer": source_counter.description,
        "target_tokenizer": target_counter.description,
        "source": summarize_lengths(source_lengths),
        "target": summarize_lengths(target_lengths),
        "target_limit": args.max_target_tokens,
        "target_at_or_below_limit": sum(length <= args.max_target_tokens for length in target_lengths),
        "target_above_limit": sum(length > args.max_target_tokens for length in target_lengths),
    }
    if not args.no_schema:
        report["fields"] = {
            field: {"count": field_counts[field], "types": sorted(field_types[field])} for field in sorted(field_counts)
        }
        report["sample_records"] = sample_records

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "length_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (args.output_dir / "lengths.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("line", "id", "source_tokens", "target_tokens"))
            writer.writeheader()
            writer.writerows(rows)
        if args.plot is None:
            args.plot = args.output_dir / "length_histogram.png"
    if args.plot:
        _plot_lengths(source_lengths, target_lengths, args.plot, args.max_target_tokens)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_dir:
        print(f"wrote report and lengths to {args.output_dir}")
    return report


def main() -> None:
    args = _build_parser().parse_args()
    try:
        inspect(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        _build_parser().error(str(exc))


if __name__ == "__main__":
    main()
