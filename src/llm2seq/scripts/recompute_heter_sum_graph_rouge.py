"""Recompute stored predictions with HeterSumGraph's ROUGE protocol.

HeterSumGraph uses ``rouge==1.0.0`` and calls
``Rouge().get_scores(hypotheses, references, avg=True)``.  Its documented data
format is already lowercase and whitespace-tokenized, so the primary score in
this script applies NFC normalization and lowercasing before the same call.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from rouge import Rouge

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = {
    "wikilingua/qwen_base": (
        "src/llm2seq/results/wikilingua/qwen/eval_base/predictions.jsonl",
        "src/llm2seq/results/wikilingua/qwen/eval_base/metrics.json",
    ),
    "wikilingua/qwen_mtp": (
        "src/llm2seq/results/wikilingua/qwen/eval_mtp/predictions.jsonl",
        "src/llm2seq/results/wikilingua/qwen/eval_mtp/metrics.json",
    ),
    "wikilingua/llama_base": (
        "src/llm2seq/results/wikilingua/llama/eval_base/predictions.jsonl",
        "src/llm2seq/results/wikilingua/llama/eval_base/metrics.json",
    ),
    "wikilingua/llama_mtp": (
        "src/llm2seq/results/wikilingua/llama/eval_mtp/predictions.jsonl",
        "src/llm2seq/results/wikilingua/llama/eval_mtp/metrics.json",
    ),
    "lrsum/qwen": (
        "src/llm2seq/results/lrsum/qwen/predictions.jsonl",
        "src/llm2seq/results/lrsum/qwen/metrics.json",
    ),
    "lrsum/llama": (
        "src/llm2seq/results/lrsum/llama/predictions.jsonl",
        "src/llm2seq/results/lrsum/llama/metrics.json",
    ),
    # Same evaluator on the external baseline is necessary for a fair gap.
    "wikilingua/t5gemma": (
        "src/T5Gemma/results/wikilingua/predictions.jsonl",
        "src/T5Gemma/results/wikilingua/metrics.json",
    ),
    "lrsum/t5gemma": (
        "src/T5Gemma/results/lrsum/predictions.jsonl",
        "src/T5Gemma/results/lrsum/metrics.json",
    ),
}


def _read_pairs(path: Path) -> Tuple[list[str], list[str]]:
    predictions = []
    references = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            predictions.append(str(row["prediction"]).strip())
            references.append(str(row["reference"]).strip())
    if not predictions:
        raise ValueError(f"No prediction/reference pairs: {path}")
    if any(not value for value in predictions + references):
        raise ValueError(f"HeterSumGraph ROUGE does not accept empty strings: {path}")
    return predictions, references


def _normalize(texts: Iterable[str]) -> list[str]:
    return [unicodedata.normalize("NFC", text).lower() for text in texts]


def _score(rouge: Rouge, predictions: list[str], references: list[str]) -> Dict[str, Dict[str, float]]:
    scores = rouge.get_scores(predictions, references, avg=True)
    return {
        name: {stat: round(100.0 * float(value), 4) for stat, value in values.items()}
        for name, values in scores.items()
    }


def recompute(output: Path, update_metrics: bool = False) -> Dict[str, Any]:
    rouge = Rouge()
    runs: Dict[str, Any] = {}
    for name, (predictions_name, metrics_name) in DEFAULT_RUNS.items():
        predictions_path = REPOSITORY_ROOT / predictions_name
        if not predictions_path.exists():
            continue
        predictions, references = _read_pairs(predictions_path)
        normalized_scores = _score(
            rouge,
            _normalize(predictions),
            _normalize(references),
        )
        runs[name] = {
            "n": len(predictions),
            "heter_sum_graph_protocol": normalized_scores,
            "predictions": predictions_name,
        }
        if update_metrics:
            metrics_path = REPOSITORY_ROOT / metrics_name
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics.update(
                    {
                        "rouge1": normalized_scores["rouge-1"]["f"],
                        "rouge2": normalized_scores["rouge-2"]["f"],
                        "rougeL": normalized_scores["rouge-l"]["f"],
                        "rougeLsum": normalized_scores["rouge-l"]["f"],
                        "rouge_backend": "rouge==1.0.0 (HeterSumGraph)",
                        "rouge_preprocessing": "NFC + lowercase + stored whitespace tokenization",
                    }
                )
                metrics_path.write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        print(
            name,
            " ".join(f"{metric}={values['f']:.4f}" for metric, values in normalized_scores.items()),
            flush=True,
        )

    result = {
        "metric": "macro-average precision/recall/F1 x 100",
        "library": f"rouge=={importlib.metadata.version('rouge')}",
        "protocol": "NFC + lowercase, then Rouge().get_scores(hyps, refs, avg=True)",
        "source": "https://github.com/dqwang122/HeterSumGraph",
        "note": "Whitespace/punctuation tokenization is kept exactly as stored in each predictions file.",
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "src/llm2seq/results/rouge_heter_sum_graph_summary.json",
    )
    parser.add_argument(
        "--update-metrics",
        action="store_true",
        help="Replace ROUGE fields in each existing metrics.json while preserving other metrics.",
    )
    args = parser.parse_args()
    recompute(args.output.expanduser().resolve(), update_metrics=args.update_metrics)
    print(f"Saved: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
