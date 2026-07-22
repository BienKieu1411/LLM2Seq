"""The exact HeterSumGraph-compatible ROUGE protocol."""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Sequence


def normalize_rouge_text(text: str, *, reference: bool = False) -> str:
    normalized = unicodedata.normalize("NFC", str(text)).lower().strip()
    return normalized or ("<empty-reference>" if reference else "<empty>")


def heter_sum_graph_rouge_per_example(
    predictions: Sequence[str],
    references: Sequence[str],
) -> List[Dict[str, float]]:
    """Return per-example F1 values under the exact stored ROUGE protocol."""

    from rouge import Rouge

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        return []
    scores = Rouge().get_scores(
        [normalize_rouge_text(text) for text in predictions],
        [normalize_rouge_text(text, reference=True) for text in references],
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


def heter_sum_graph_rouge(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    include_rouge_lsum: bool = True,
    digits: int = 4,
) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        result = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        if include_rouge_lsum:
            result["rougeLsum"] = 0.0
        return result

    per_example = heter_sum_graph_rouge_per_example(predictions, references)
    result = {
        metric: round(sum(score[metric] for score in per_example) / len(per_example), digits)
        for metric in ("rouge1", "rouge2", "rougeL")
    }
    if include_rouge_lsum:
        result["rougeLsum"] = result["rougeL"]
    return result
