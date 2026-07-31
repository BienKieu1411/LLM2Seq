"""Built-in text-to-text metrics."""

from __future__ import annotations

import unicodedata
import re
from importlib import import_module
from collections import Counter
from typing import Any, Dict, List, Sequence


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
        name: round(sum(item[name] for item in scores) / len(scores), 4) for name in ("rouge1", "rouge2", "rougeL")
    }
    result["rougeLsum"] = result["rougeL"]
    return result


def exact_match_score(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    matches = sum(
        _normalize(prediction) == _normalize(reference, reference=True)
        for prediction, reference in zip(predictions, references)
    )
    return {"exact_match": round(100.0 * matches / max(1, len(predictions)), 4)}


def _metric_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize(text), flags=re.UNICODE)


def token_f1_score(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    values: List[float] = []
    for prediction, reference in zip(predictions, references):
        predicted = Counter(_metric_tokens(prediction))
        expected = Counter(_metric_tokens(reference))
        overlap = sum((predicted & expected).values())
        if not predicted and not expected:
            values.append(1.0)
        elif overlap == 0:
            values.append(0.0)
        else:
            precision = overlap / sum(predicted.values())
            recall = overlap / sum(expected.values())
            values.append(2.0 * precision * recall / (precision + recall))
    return {"token_f1": round(100.0 * sum(values) / max(1, len(values)), 4)}


def task_scores(
    predictions: Sequence[str],
    references: Sequence[str],
    task_config: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Evaluate a configured text-to-text task with built-in metrics."""

    configured = (task_config or {}).get("metrics", ["rouge"])
    result: Dict[str, float] = {}
    for metric in configured:
        if metric == "rouge":
            result.update(rouge_scores(predictions, references))
        elif metric == "exact_match":
            result.update(exact_match_score(predictions, references))
        elif metric == "token_f1":
            result.update(token_f1_score(predictions, references))
        else:
            raise ValueError(f"Unsupported task metric: {metric}")
    callable_path = str((task_config or {}).get("metric_callable", "")).strip()
    if callable_path:
        module_name, function_name = callable_path.split(":", 1)
        function = getattr(import_module(module_name), function_name)
        custom = function(list(predictions), list(references))
        if not isinstance(custom, dict):
            raise TypeError("Custom task metric must return a dictionary")
        result.update({str(name): float(value) for name, value in custom.items()})
    return result
