"""Filter a summarization JSONL file by target length and optionally take N samples."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from jsonl_length_utils import load_counters, record_texts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("output", type=Path, help="Filtered JSONL output; must differ from input")
    parser.add_argument("--tokenizer", help="Local tokenizer used for target length")
    parser.add_argument("--target-tokenizer", help="Local target tokenizer (overrides --tokenizer)")
    parser.add_argument("--source-field", default="text")
    parser.add_argument("--target-field", default="summary")
    parser.add_argument("--list-separator", default="\n")
    bounds = parser.add_mutually_exclusive_group()
    bounds.add_argument("--max-target-tokens", type=int, help="Inclusive upper bound (target length <= N)")
    bounds.add_argument("--target-below", type=int, help="Strict upper bound (target length < N), e.g. 512")
    parser.add_argument("--min-target-tokens", type=int, default=0, help="Inclusive lower bound")
    parser.add_argument("--num-samples", type=int, help="Keep exactly N eligible samples")
    parser.add_argument("--selection", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--allow-fewer", action="store_true", help="Write all eligible rows if fewer than --num-samples exist"
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face tokenizer downloads")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _eligible(length: int, args: argparse.Namespace) -> bool:
    if length < args.min_target_tokens:
        return False
    if args.max_target_tokens is not None and length > args.max_target_tokens:
        return False
    if args.target_below is not None and length >= args.target_below:
        return False
    return True


def filter_jsonl(args: argparse.Namespace) -> dict:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path == output_path:
        raise ValueError("input and output must be different paths")
    if (
        args.min_target_tokens < 0
        or (args.max_target_tokens is not None and args.max_target_tokens < 0)
        or (args.target_below is not None and args.target_below <= 0)
    ):
        raise ValueError("target length bounds must be non-negative")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    _source_counter, target_counter = load_counters(
        tokenizer=args.tokenizer,
        source_tokenizer=None,
        target_tokenizer=args.target_tokenizer,
        allow_download=args.allow_download,
        trust_remote_code=args.trust_remote_code,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)[1])
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
                    _source, target = record_texts(
                        row,
                        source_field=args.source_field,
                        target_field=args.target_field,
                        separator=args.list_separator,
                    )
                    if not target:
                        raise ValueError("target is empty")
                    target_length = target_counter(target)
                except (json.JSONDecodeError, TypeError, ValueError):
                    invalid_count += 1
                    continue
                if not _eligible(target_length, args):
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
        "tokenizer": target_counter.description,
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
