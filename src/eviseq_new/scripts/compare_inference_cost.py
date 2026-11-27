from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.append(str(Path(__file__).resolve().parent))

from inference_cost import markdown_table, read_rows


def _workload(row: dict[str, Any]) -> tuple:
    return row["system"], row["source_tokens"], row["batch_size"]


def _total(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing"
    return f"{row['total_s']:.2f}" if row["status"] == "ok" else "OOM"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check one or two inference cost passes against each other.")
    parser.add_argument("passes", nargs="+", type=Path, help="JSONL file of one pass; give the second to compare")
    parser.add_argument("--expect", nargs="+", required=True, help="systems every pass should contain")
    parser.add_argument("--tolerance", type=float, default=5.0, help="allowed difference in total time, percent")
    args = parser.parse_args(argv)
    if len(args.passes) > 2:
        parser.error("give one or two passes")

    passes = [read_rows(path) for path in args.passes]
    everything = [row for rows in passes for row in rows]
    environments = sorted(
        {tuple(str(row.get(name)) for name in ("gpu", "torch", "transformers", "dtype")) for row in everything}
    )
    for gpu, torch_version, transformers_version, dtype in environments:
        print(f"[env] {gpu} | torch {torch_version} | transformers {transformers_version} | {dtype}")

    order = {name: index for index, name in enumerate(args.expect)}

    def position(key: tuple) -> tuple:
        return order.get(key[0], len(order)), key[1], key[2]

    if passes[0]:
        print(markdown_table(sorted(passes[0], key=lambda row: position(_workload(row)))))

    if len(passes) == 2:
        first, second = ({_workload(row): row for row in rows} for rows in passes)
        print("\n| system | source tokens | batch | total A (s) | total B (s) | B vs A (%) |")
        print("|---|---|---|---|---|---|")
        compared = differing = 0
        for key in sorted({*first, *second}, key=position):
            a, b = first.get(key), second.get(key)
            change = "-"
            if a is not None and b is not None and a["status"] == b["status"] == "ok":
                percent = (b["total_s"] / a["total_s"] - 1) * 100
                change = f"{percent:+.1f}"
                compared += 1
                differing += abs(percent) > args.tolerance
            print("| " + " | ".join([*(str(part) for part in key), _total(a), _total(b), change]) + " |")
        print(
            f"[check] {differing} of {compared} workloads differ by more than {args.tolerance:g}% "
            "between the two passes"
        )

    workloads = sorted({_workload(row)[1:] for row in everything})
    if not workloads:
        print("[check] no rows were measured")
    for path, rows in zip(args.passes, passes):
        measured = {_workload(row) for row in rows}
        for system in args.expect:
            absent = [f"{length}/{batch}" for length, batch in workloads if (system, length, batch) not in measured]
            if absent or not workloads:
                print(f"[check] {path.name} is missing {system} {', '.join(absent)}".rstrip())
    if len(environments) > 1:
        print(f"[check] rows come from {len(environments)} environments; do not compare them")


if __name__ == "__main__":
    main()
