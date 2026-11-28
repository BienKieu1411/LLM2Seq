from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT.parent
sys.path.append(str(Path(__file__).resolve().parent))

from export_samples import joined, normalize, read_rows

ROUGE_HOME = "/workspace/storage-shared/nlp/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5"
METRICS = ("rouge1", "rouge2", "rougeL")


def align(systems: dict[str, Path], candidate: str) -> dict[str, list[dict[str, str]]]:
    base = [
        {"id": str(row.get("id") or f"row-{index + 1:08d}"), "reference": joined(row["reference"]),
         "prediction": row["prediction"]}
        for index, row in enumerate(read_rows(systems[candidate]))
    ]
    if len({row["id"] for row in base}) != len(base):
        raise SystemExit(f"{candidate} repeats an example id")
    aligned = {candidate: base}
    for label, path in systems.items():
        if label == candidate:
            continue
        waiting: dict[str, list[str]] = {}
        for row in read_rows(path):
            waiting.setdefault(normalize(joined(row["reference"])), []).append(row["prediction"])
        rows = []
        for row in base:
            queue = waiting.get(normalize(row["reference"]))
            if queue:
                rows.append({**row, "prediction": queue.pop(0)})
        if len(rows) != len(base):
            raise SystemExit(
                f"{label} has no prediction for {len(base) - len(rows)} of the {len(base)} examples of {candidate}"
            )
        aligned[label] = rows
    return aligned


def _name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Official ROUGE for every system and paired bootstrap against one.")
    parser.add_argument("--candidate", required=True, metavar="LABEL=PREDICTIONS.jsonl")
    parser.add_argument("--baseline", action="append", required=True, metavar="LABEL=PREDICTIONS.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "runs/rebuttal/bootstrap")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    systems: dict[str, Path] = {}
    for value in (args.candidate, *args.baseline):
        label, separator, path = value.partition("=")
        if not separator or not label or not Path(path).is_file() or label in systems:
            raise SystemExit(f"Expected a new LABEL=PATH of an existing file, got {value!r}")
        systems[label] = Path(path)
    candidate = args.candidate.partition("=")[0]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONPATH": str(SOURCE_ROOT)}
    if Path(ROUGE_HOME).is_dir():
        environment.setdefault("PYROUGE_HOME_DIR", ROUGE_HOME)

    scores, changed = {}, set()
    for label, rows in align(systems, candidate).items():
        predictions = out / f"{_name(label)}.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        result = out / f"{_name(label)}.rouge.json"
        if not predictions.exists() or predictions.read_text(encoding="utf-8") != text:
            predictions.write_text(text, encoding="utf-8")
            result.unlink(missing_ok=True)
            changed.add(label)
        if not result.exists():
            print(f"=== ROUGE: {label}", flush=True)
            subprocess.run(
                [sys.executable, str(SOURCE_ROOT / "rouge155/evaluate_rouge.py"), str(predictions), "--details",
                 "--output", str(result)], check=True, env=environment)
        scores[label] = result

    lines = [f"| system | ROUGE-1 | ROUGE-2 | ROUGE-L | {candidate} minus system, ROUGE-1 [95% CI] "
             "| ROUGE-2 [95% CI] | ROUGE-L [95% CI] |", "|---|---|---|---|---|---|---|"]
    examples = None
    for label, result in scores.items():
        headline = json.loads(result.read_text(encoding="utf-8"))
        cells = [label, *(f"{headline[metric]:.2f}" for metric in METRICS)]
        if label == candidate:
            cells += ["", "", ""]
        else:
            output = out / f"{_name(candidate)}_vs_{_name(label)}.bootstrap.json"
            if changed & {candidate, label}:
                output.unlink(missing_ok=True)
            if not output.exists():
                print(f"=== paired bootstrap: {candidate} vs {label}", flush=True)
                subprocess.run(
                    [sys.executable, "-m", "evaluation.paired_bootstrap", str(scores[candidate]), str(result),
                     "--output", str(output), "--samples", str(args.samples), "--seed", str(args.seed)],
                    check=True, env=environment)
            comparison = json.loads(output.read_text(encoding="utf-8"))
            examples = comparison["num_examples"]
            cells += [
                "{paired_mean_delta:+.2f} [{ci95_low:+.2f}, {ci95_high:+.2f}]".format(**comparison["scores"][metric])
                for metric in METRICS
            ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"Paired bootstrap over {examples} test examples, {args.samples} resamples, seed {args.seed}."]
    (out / "bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
