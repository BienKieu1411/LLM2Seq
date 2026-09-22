"""Filter a summarization JSONL file by source/target length and take N samples."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
from jsonl_length_utils import load_counters, record_texts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("output", type=Path, help="Filtered JSONL output; must differ from input")
    parser.add_argument("--tokenizer", help="One local tokenizer used for source and target lengths")
    parser.add_argument("--source-tokenizer", help="Local tokenizer used for source length")
    parser.add_argument("--target-tokenizer", help="Local target tokenizer (overrides --tokenizer)")
    parser.add_argument("--source-field", default="text")
    parser.add_argument("--target-field", default="summary")
    parser.add_argument("--list-separator", default="\n")
    source_bounds = parser.add_mutually_exclusive_group()
    source_bounds.add_argument("--max-source-tokens", type=int, help="Inclusive upper bound (source length <= N)")
    source_bounds.add_argument("--source-below", type=int, help="Strict upper bound (source length < N)")
    source_bounds.add_argument("--min-source-tokens", type=int, default=0, help="Inclusive source lower bound")
    target_bounds = parser.add_mutually_exclusive_group()
    target_bounds.add_argument("--max-target-tokens", type=int, help="Inclusive upper bound (target length <= N)")
    target_bounds.add_argument("--target-below", type=int, help="Strict upper bound (target length < N), e.g. 512")
    parser.add_argument("--min-target-tokens", type=int, default=0, help="Inclusive lower bound")
    parser.add_argument("--num-samples", type=int, help="Keep exactly N eligible samples")
    parser.add_argument(
        "--balance-source-bins",
        type=int,
        help="Split eligible source lengths into this many equal-width bins before applying a per-bin cap",
    )
    parser.add_argument(
        "--max-per-source-bin",
        type=int,
        default=20_000,
        help="Maximum samples retained from each source-length bin (default: 20000)",
    )
    parser.add_argument("--selection", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--allow-fewer", action="store_true", help="Write all eligible rows if fewer than --num-samples exist"
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face tokenizer downloads")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _eligible(length: int, *, minimum: int, maximum: int | None, strict_upper: int | None) -> bool:
    if length < minimum:
        return False
    if maximum is not None and length > maximum:
        return False
    if strict_upper is not None and length >= strict_upper:
        return False
    return True


def _measure_line(
    raw_line: str,
    *,
    source_field: str,
    target_field: str,
    list_separator: str,
    source_counter,
    target_counter,
) -> tuple[int, int]:
    row = json.loads(raw_line)
    if not isinstance(row, dict):
        raise ValueError("record is not an object")
    source, target = record_texts(
        row,
        source_field=source_field,
        target_field=target_field,
        separator=list_separator,
    )
    if not source or not target:
        raise ValueError("source and target must be non-empty")
    return source_counter(source), target_counter(target)


def _collect_eligible(
    args: argparse.Namespace,
    *,
    source_counter,
    target_counter,
) -> tuple[list[tuple[int, int, int]], int]:
    """Collect only line numbers and lengths; raw documents stay out of memory."""

    records: list[tuple[int, int, int]] = []
    invalid_count = 0
    with args.input.open("r", encoding="utf-8-sig") as source_handle:
        for line_number, raw_line in enumerate(source_handle, start=1):
            if not raw_line.strip():
                continue
            try:
                source_length, target_length = _measure_line(
                    raw_line,
                    source_field=args.source_field,
                    target_field=args.target_field,
                    list_separator=args.list_separator,
                    source_counter=source_counter,
                    target_counter=target_counter,
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                invalid_count += 1
                continue
            source_ok = _eligible(
                source_length,
                minimum=args.min_source_tokens,
                maximum=args.max_source_tokens,
                strict_upper=args.source_below,
            )
            target_ok = _eligible(
                target_length,
                minimum=args.min_target_tokens,
                maximum=args.max_target_tokens,
                strict_upper=args.target_below,
            )
            if source_ok and target_ok:
                records.append((line_number, source_length, target_length))
    return records, invalid_count


def _write_selected_lines(input_path: Path, output_path: Path, selected_lines: set[int]) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with (
            input_path.open("r", encoding="utf-8-sig") as source_handle,
            temporary.open("w", encoding="utf-8") as output_handle,
        ):
            for line_number, raw_line in enumerate(source_handle, start=1):
                if line_number in selected_lines:
                    output_handle.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _balance_source_bins(
    records: list[tuple[int, int, int]],
    *,
    number_of_bins: int,
    maximum_per_bin: int,
    selection: str,
    seed: int,
) -> tuple[set[int], dict]:
    source_values = np.asarray([record[1] for record in records], dtype=np.int64)
    counts, edges = np.histogram(source_values, bins=number_of_bins)
    bin_indices = np.searchsorted(edges, source_values, side="right") - 1
    bin_indices = np.minimum(np.maximum(bin_indices, 0), number_of_bins - 1)
    chosen: list[list[int]] = [[] for _ in range(number_of_bins)]
    seen = [0] * number_of_bins
    rng = random.Random(seed)
    for record, bin_index_value in zip(records, bin_indices.tolist()):
        bin_index = int(bin_index_value)
        seen[bin_index] += 1
        if len(chosen[bin_index]) < maximum_per_bin:
            chosen[bin_index].append(record[0])
        elif selection == "random":
            replacement = rng.randrange(seen[bin_index])
            if replacement < maximum_per_bin:
                chosen[bin_index][replacement] = record[0]
    selected_lines = {line_number for bin_lines in chosen for line_number in bin_lines}
    bin_report = []
    for index, (left, right, before) in enumerate(zip(edges[:-1], edges[1:], counts.tolist())):
        bin_report.append(
            {
                "bin": index,
                "source_token_start_inclusive": int(np.ceil(left)),
                "source_token_end_inclusive": int(np.ceil(right)) - 1,
                "eligible_samples": int(before),
                "written_samples": len(chosen[index]),
            }
        )
    return selected_lines, {
        "number_of_bins": number_of_bins,
        "max_per_bin": maximum_per_bin,
        "eligible_samples": len(records),
        "written_samples": len(selected_lines),
        "bins": bin_report,
    }


def filter_jsonl(args: argparse.Namespace) -> dict:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path == output_path:
        raise ValueError("input and output must be different paths")
    invalid_bounds = (
        args.min_source_tokens < 0
        or args.min_target_tokens < 0
        or (args.max_source_tokens is not None and args.max_source_tokens < 0)
        or (args.source_below is not None and args.source_below <= 0)
        or (args.max_target_tokens is not None and args.max_target_tokens < 0)
        or (args.target_below is not None and args.target_below <= 0)
    )
    if invalid_bounds:
        raise ValueError("source/target length bounds must be non-negative")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.balance_source_bins is not None and args.balance_source_bins <= 0:
        raise ValueError("--balance-source-bins must be positive")
    if args.max_per_source_bin <= 0:
        raise ValueError("--max-per-source-bin must be positive")
    if args.balance_source_bins is not None and args.num_samples is not None:
        raise ValueError("use either --num-samples or --balance-source-bins, not both")

    source_counter, target_counter = load_counters(
        tokenizer=args.tokenizer,
        source_tokenizer=args.source_tokenizer,
        target_tokenizer=args.target_tokenizer,
        allow_download=args.allow_download,
        trust_remote_code=args.trust_remote_code,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.balance_source_bins is not None:
        eligible_records, invalid_count = _collect_eligible(
            args,
            source_counter=source_counter,
            target_counter=target_counter,
        )
        if not eligible_records:
            raise ValueError("no samples satisfy the source/target length filters")
        selected_lines, balance_report = _balance_source_bins(
            eligible_records,
            number_of_bins=args.balance_source_bins,
            maximum_per_bin=args.max_per_source_bin,
            selection=args.selection,
            seed=args.seed,
        )
        _write_selected_lines(input_path, output_path, selected_lines)
        report = {
            "input": str(input_path),
            "output": str(output_path),
            "source_tokenizer": source_counter.description,
            "target_tokenizer": target_counter.description,
            "min_source_tokens": args.min_source_tokens,
            "max_source_tokens": args.max_source_tokens,
            "source_below": args.source_below,
            "min_target_tokens": args.min_target_tokens,
            "max_target_tokens": args.max_target_tokens,
            "target_below": args.target_below,
            "eligible_samples": len(eligible_records),
            "written_samples": len(selected_lines),
            "invalid_rows": invalid_count,
            "selection": args.selection,
            "seed": args.seed,
            "source_balance": balance_report,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    eligible_count = 0
    invalid_count = 0
    seen = 0
    selected: list[tuple[int, str]] = []
    rng = random.Random(args.seed)
    try:
        with (
            input_path.open("r", encoding="utf-8-sig") as source_handle,
            temporary.open("w", encoding="utf-8") as first_output,
        ):
            for line_number, raw_line in enumerate(source_handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                    if not isinstance(row, dict):
                        raise ValueError("record is not an object")
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
                except (json.JSONDecodeError, TypeError, ValueError):
                    invalid_count += 1
                    continue
                source_ok = _eligible(
                    source_length,
                    minimum=args.min_source_tokens,
                    maximum=args.max_source_tokens,
                    strict_upper=args.source_below,
                )
                target_ok = _eligible(
                    target_length,
                    minimum=args.min_target_tokens,
                    maximum=args.max_target_tokens,
                    strict_upper=args.target_below,
                )
                if not source_ok or not target_ok:
                    continue
                eligible_count += 1
                if args.num_samples is None:
                    first_output.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                    seen += 1
                    continue
                if args.selection == "first":
                    if seen < args.num_samples:
                        first_output.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                        seen += 1
                else:
                    # Reservoir sampling stores only N long JSON lines in memory.
                    if len(selected) < args.num_samples:
                        selected.append((line_number, raw_line))
                    else:
                        replacement = rng.randrange(eligible_count)
                        if replacement < args.num_samples:
                            selected[replacement] = (line_number, raw_line)
            if args.num_samples is not None and args.selection == "random":
                first_output.seek(0)
                first_output.truncate(0)
                for _line_number, raw_line in sorted(selected):
                    first_output.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")

        if args.num_samples is not None and eligible_count < args.num_samples and not args.allow_fewer:
            raise ValueError(
                f"only {eligible_count} eligible samples satisfy the length filter, "
                f"but --num-samples requested {args.num_samples}; use --allow-fewer to continue"
            )
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    written = min(eligible_count, args.num_samples) if args.num_samples is not None else eligible_count
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "source_tokenizer": source_counter.description,
        "target_tokenizer": target_counter.description,
        "min_source_tokens": args.min_source_tokens,
        "max_source_tokens": args.max_source_tokens,
        "source_below": args.source_below,
        "min_target_tokens": args.min_target_tokens,
        "max_target_tokens": args.max_target_tokens,
        "target_below": args.target_below,
        "eligible_samples": eligible_count,
        "written_samples": written,
        "invalid_rows": invalid_count,
        "selection": args.selection if args.num_samples is not None else "all",
        "seed": args.seed,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    args = _build_parser().parse_args()
    try:
        filter_jsonl(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        _build_parser().error(str(exc))


if __name__ == "__main__":
    main()
