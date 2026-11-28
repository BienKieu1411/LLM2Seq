from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(Path(__file__).resolve().parent))

from export_samples import joined, normalize, read_rows


def load_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def truncation_stats(texts: Sequence[str], tokenizer: Any, limit: int) -> dict[str, Any]:
    counts = [len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts]
    return {
        "texts": len(counts),
        "truncated_pct": 100 * sum(count > limit for count in counts) / len(counts),
        "unscored_tokens_pct": 100 * sum(max(0, count - limit) for count in counts) / sum(counts),
        "mean_tokens": sum(counts) / len(counts),
        "max_tokens": max(counts),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="How many summaries BERTScore cuts at its tokenizer limit.")
    parser.add_argument("--system", action="append", required=True, metavar="LABEL=PREDICTIONS.jsonl")
    parser.add_argument("--bertscore-json", type=Path, help="a finished .bertscore.json; gives model and limit")
    parser.add_argument("--tokenizer", help="BERTScore model folder, if no .bertscore.json is given")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "runs/rebuttal/bertscore_truncation.md")
    args = parser.parse_args(argv)
    recorded = json.loads(args.bertscore_json.read_text(encoding="utf-8")) if args.bertscore_json else {}
    model = args.tokenizer or recorded.get("model_path")
    limit = args.max_length or recorded.get("tokenizer_max_length")
    if not model or not limit:
        raise SystemExit("Give --bertscore-json, or both --tokenizer and --max-length")
    tokenizer = load_tokenizer(str(model))
    rows, references = {}, None
    for value in args.system:
        label, separator, path = value.partition("=")
        if not separator or not Path(path).is_file():
            raise SystemExit(f"--system expects LABEL=PATH of an existing file, got {value!r}")
        records = read_rows(Path(path))
        keys = sorted(normalize(joined(record["reference"])) for record in records)
        if references is None:
            references, first = keys, records
        elif keys != references:
            raise SystemExit(f"{label} was scored against a different reference set")
        rows[label] = truncation_stats([record["prediction"] for record in records], tokenizer, int(limit))
    table = {"reference summaries": truncation_stats([joined(r["reference"]) for r in first], tokenizer, int(limit)),
             **rows}
    lines = [
        f"BERTScore model {model}, limit {limit} tokens", "",
        "| summaries | texts | over the limit (%) | tokens never scored (%) | mean tokens | max tokens |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {name} | {row['texts']} | {row['truncated_pct']:.1f} | {row['unscored_tokens_pct']:.1f} "
        f"| {row['mean_tokens']:.0f} | {row['max_tokens']} |"
        for name, row in table.items()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
