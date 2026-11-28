from __future__ import annotations

import argparse
import csv
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(Path(__file__).resolve().parent))

from export_samples import DEFAULT_SOURCE, matched_documents

CRITERIA = ("faithfulness", "coverage", "readability")
CATEGORIES = 4
MIN_EXAMPLES_FOR_INTERVAL = 30
LABELS = "ABCDEFGH"
SHEET_COLUMNS = ("sample_code", "label", "summary", *CRITERIA, "notes")
KEY_COLUMNS = ("sample_code", "real_id", "label", "system")


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), restval="")
        writer.writeheader()
        writer.writerows(rows)


def make_sheet(args: argparse.Namespace) -> None:
    systems = {}
    for value in args.system:
        label, separator, path = value.partition("=")
        if not separator or not label or not Path(path).is_file():
            raise SystemExit(f"--system expects LABEL=PATH of an existing file, got {value!r}")
        systems[label] = Path(path)
    if not 2 <= len(systems) <= len(LABELS):
        raise SystemExit(f"Give between 2 and {len(LABELS)} systems")
    matched = matched_documents(args.source, systems)
    if len(matched) < args.n:
        raise SystemExit(f"only {len(matched)} documents have a summary from every system; {args.n} needed")
    generator = random.Random(args.seed)
    sheet, key = [], []
    (args.output_dir / "sources").mkdir(parents=True, exist_ok=True)
    for position, (document, summaries) in enumerate(generator.sample(matched, args.n), start=1):
        code = f"S{position:03d}"
        order = list(systems)
        generator.shuffle(order)
        (args.output_dir / "sources" / f"{code}.txt").write_text(document["source"], encoding="utf-8")
        for label, system in zip(LABELS, order):
            sheet.append({"sample_code": code, "label": label, "summary": summaries[system]})
            key.append({"sample_code": code, "real_id": document["id"], "label": label, "system": system})
    _write_csv(args.output_dir / "sheet.csv", SHEET_COLUMNS, sheet)
    _write_csv(args.output_dir / "key.csv", KEY_COLUMNS, key)
    print(f"{args.n} documents x {len(systems)} systems -> {args.output_dir}")
    print("Give annotators sheet.csv and sources/; keep key.csv to yourself.")


def load_annotations(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in ("sample_code", "label", *CRITERIA) if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} lacks columns: {missing}")
        rows = []
        for line, record in enumerate(reader, start=2):
            row: dict[str, Any] = {"sample_code": record["sample_code"].strip(), "label": record["label"].strip()}
            for criterion in CRITERIA:
                text = (record[criterion] or "").strip()
                if text not in {str(value) for value in range(1, CATEGORIES + 1)}:
                    raise ValueError(f"{path}, line {line}: {criterion} is {text!r}, expected 1 to {CATEGORIES}")
                row[criterion] = int(text)
            rows.append(row)
    return rows


def weighted_kappa(a: Sequence[int], b: Sequence[int]) -> float | None:
    observed = np.zeros((CATEGORIES, CATEGORIES))
    for x, y in zip(a, b):
        observed[x - 1, y - 1] += 1
    index = np.arange(CATEGORIES)
    weight = (index[:, None] - index[None, :]) ** 2 / (CATEGORIES - 1) ** 2
    expected = np.outer(observed.sum(1), observed.sum(0)) / len(a)
    chance = float((weight * expected).sum())
    return None if chance == 0 else 1.0 - float((weight * observed).sum()) / chance


def krippendorff_alpha(units: Sequence[Sequence[int]]) -> float | None:
    values = np.asarray([value for unit in units for value in unit], dtype=float)
    total = values.size
    observed = sum(((np.subtract.outer(unit, unit)) ** 2).sum() / (len(unit) - 1) for unit in units)
    expected = (2 * total * (values ** 2).sum() - 2 * values.sum() ** 2)
    if total < 2 or expected <= 0:
        return None
    return float(1.0 - (observed / total) / (expected / (total * (total - 1))))


def _interval(values: np.ndarray, samples: int, seed: int) -> str:
    if values.size < MIN_EXAMPLES_FOR_INTERVAL:
        return ""
    generator = np.random.default_rng(seed)
    means = values[generator.integers(0, values.size, size=(samples, values.size))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return f" [{low:+.2f}, {high:+.2f}]"


def score(args: argparse.Namespace) -> None:
    with args.key.open("r", encoding="utf-8", newline="") as handle:
        key_rows = list(csv.DictReader(handle))
    key = {(row["sample_code"], row["label"]): row["system"] for row in key_rows}
    systems = list(dict.fromkeys(row["system"] for row in key_rows))
    codes = list(dict.fromkeys(row["sample_code"] for row in key_rows))
    if args.reference not in systems:
        raise SystemExit(f"--reference must be one of {systems}")
    annotators = dict(value.split("=", 1) for value in args.annotator)
    if len(annotators) < 2:
        raise SystemExit("Give at least two --annotator NAME=FILE to measure agreement")
    ratings = {}
    for name, path in annotators.items():
        table = {(row["sample_code"], row["label"]): row for row in load_annotations(Path(path))}
        if set(table) != set(key):
            raise SystemExit(f"{name}: the sheet does not rate exactly the rows of the key")
        ratings[name] = table
    cell_of = {(code, system): (code, label) for (code, label), system in key.items()}
    cells = sorted(key)
    lines = [f"{len(codes)} samples, annotators: {', '.join(ratings)}", ""]
    for criterion in CRITERIA:
        columns = {name: [ratings[name][cell][criterion] for cell in cells] for name in ratings}
        pairs = list(combinations(ratings, 2))
        agreement = [(x, y) for a, b in pairs for x, y in zip(columns[a], columns[b])]
        alpha = krippendorff_alpha([[columns[name][i] for name in ratings] for i in range(len(cells))])
        kappas = ", ".join(
            f"{a}|{b} {value:.2f}" for a, b in pairs if (value := weighted_kappa(columns[a], columns[b])) is not None
        )
        lines += [
            f"### {criterion}", "",
            f"Krippendorff alpha {'-' if alpha is None else f'{alpha:.2f}'}; exact agreement "
            f"{sum(x == y for x, y in agreement) / len(agreement):.2f}; within one point "
            f"{sum(abs(x - y) <= 1 for x, y in agreement) / len(agreement):.2f}; weighted kappa {kappas}", "",
            f"| system | mean (1-{CATEGORIES}) | {args.reference} minus system (95% CI) |", "|---|---|---|",
        ]
        means = {
            system: np.asarray([np.mean([ratings[name][cell_of[(code, system)]][criterion] for name in ratings])
                                for code in codes])
            for system in systems
        }
        for system in systems:
            difference = means[args.reference] - means[system]
            versus = "" if system == args.reference else (
                f"{difference.mean():+.2f}{_interval(difference, args.samples, args.seed)}")
            lines.append(f"| {system} | {means[system].mean():.2f} | {versus} |")
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Blind human-evaluation sheet and its scores.")
    commands = parser.add_subparsers(dest="command", required=True)
    sheet = commands.add_parser("sheet", help="write a blind, shuffled rating sheet")
    sheet.add_argument("--system", action="append", required=True, metavar="LABEL=PREDICTIONS.jsonl")
    sheet.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    sheet.add_argument("--n", type=int, default=30)
    sheet.add_argument("--seed", type=int, default=1729)
    sheet.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "runs/rebuttal/human_eval")
    results = commands.add_parser("score", help="agreement and system scores from the filled sheets")
    results.add_argument("--key", type=Path, required=True)
    results.add_argument("--annotator", action="append", required=True, metavar="NAME=FILE.csv")
    results.add_argument("--reference", required=True)
    results.add_argument("--samples", type=int, default=10_000)
    results.add_argument("--seed", type=int, default=1729)
    results.add_argument("--output", type=Path, default=PACKAGE_ROOT / "runs/rebuttal/human_eval/report.md")
    args = parser.parse_args(argv)
    make_sheet(args) if args.command == "sheet" else score(args)


if __name__ == "__main__":
    main()
