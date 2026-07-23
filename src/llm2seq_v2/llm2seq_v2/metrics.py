"""Exact rouge==1.0.0 protocol used by the local T5Gemma comparison."""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Sequence


def _normalize(text: str, reference: bool = False) -> str:
    value = unicodedata.normalize("NFC", str(text)).lower().strip()
    return value or ("<empty-reference>" if reference else "<empty>")


def rouge_per_example(predictions: Sequence[str], references: Sequence[str]) -> List[Dict[str, float]]:
    from rouge import Rouge

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        return []
    scores = Rouge().get_scores(
        [_normalize(value) for value in predictions],
        [_normalize(value, reference=True) for value in references],
        avg=False,
    )
    return [
        {
            "rouge1": 100.0 * float(score["rouge-1"]["f"]),
            "rouge2": 100.0 * float(score["rouge-2"]["f"]),
            "rougeL": 100.0 * float(score["rouge-l"]["f"]),
        }
        for score in scores
    ]


def rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    scores = rouge_per_example(predictions, references)
    if not scores:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    result = {
        name: round(sum(item[name] for item in scores) / len(scores), 4)
        for name in ("rouge1", "rouge2", "rougeL")
    }
    result["rougeLsum"] = result["rougeL"]
    return result
