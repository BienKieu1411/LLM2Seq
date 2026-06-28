#!/usr/bin/env python3
"""Compute BERTScore for the WikiLingua prediction files used in the report.

The script writes a dataset-level BERTScore summary and also injects the
corresponding BERTScore fields into each system's metrics.json file.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from bert_score import score


ROOT = Path(__file__).resolve().parents[3]

SYSTEMS = {
    "llama_ar": {
        "predictions": ROOT / "src/llm2seq/results/wikilingua/llama/eval_base/predictions.jsonl",
        "metrics": ROOT / "src/llm2seq/results/wikilingua/llama/eval_base/metrics.json",
    },
    "llama_mtp": {
        "predictions": ROOT / "src/llm2seq/results/wikilingua/llama/eval_mtp/predictions.jsonl",
        "metrics": ROOT / "src/llm2seq/results/wikilingua/llama/eval_mtp/metrics.json",
    },
    "qwen_ar": {
        "predictions": ROOT / "src/llm2seq/results/wikilingua/qwen/eval_base/predictions.jsonl",
        "metrics": ROOT / "src/llm2seq/results/wikilingua/qwen/eval_base/metrics.json",
    },
    "qwen_mtp": {
        "predictions": ROOT / "src/llm2seq/results/wikilingua/qwen/eval_mtp/predictions.jsonl",
        "metrics": ROOT / "src/llm2seq/results/wikilingua/qwen/eval_mtp/metrics.json",
    },
    "t5gemma": {
        "predictions": ROOT / "src/T5Gemma/results/wikilingua/predictions.jsonl",
        "metrics": ROOT / "src/T5Gemma/results/wikilingua/metrics.json",
    },
}


def read_predictions(path: Path) -> tuple[list[str], list[str]]:
    predictions: list[str] = []
    references: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            predictions.append(row["prediction"])
            references.append(row["reference"])
    return predictions, references


def update_metrics_file(path: Path, values: dict[str, float | int | str | bool]) -> None:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics.update(values)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_type = "bert-base-multilingual-cased"
    output_path = ROOT / "src/llm2seq/results/wikilingua/bertscore_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metric": "BERTScore",
        "model_type": model_type,
        "device": device,
        "rescale_with_baseline": False,
        "scoring_direction": "prediction_vs_reference",
        "systems": {},
    }

    for name, paths in SYSTEMS.items():
        predictions_path = paths["predictions"]
        metrics_path = paths["metrics"]
        predictions, references = read_predictions(predictions_path)
        precision, recall, f1 = score(
            predictions,
            references,
            model_type=model_type,
            lang="vi",
            device=device,
            batch_size=16,
            verbose=True,
            rescale_with_baseline=False,
        )
        system_scores = {
            "num_examples": len(predictions),
            "precision": round(100.0 * precision.mean().item(), 4),
            "recall": round(100.0 * recall.mean().item(), 4),
            "f1": round(100.0 * f1.mean().item(), 4),
            "predictions_file": str(predictions_path.relative_to(ROOT)),
        }
        results["systems"][name] = system_scores
        update_metrics_file(
            metrics_path,
            {
                "bertscore_precision": system_scores["precision"],
                "bertscore_recall": system_scores["recall"],
                "bertscore_f1": system_scores["f1"],
                "bertscore_model_type": model_type,
                "bertscore_rescale_with_baseline": False,
            },
        )
        print(name, results["systems"][name])

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
