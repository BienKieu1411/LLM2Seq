"""Fast, Unicode-preserving ROUGE proxy for one generated summary/reference pair.

This scorer is deliberately dependency-free. It is a training proxy, not a
replacement for the project's Perl ROUGE-1.5.5 evaluation protocol.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from numbers import Real
from typing import Any

_PREAMBLE = "Here is a summary of the text in Vietnamese"
_LEGACY_REMAP = {
    "-lrb-": "(",
    "-rrb-": ")",
    "-lcb-": "{",
    "-rcb-": "}",
    "-lsb-": "[",
    "-rsb-": "]",
    "``": '"',
    "''": '"',
}
_LEGACY_REMAP_RE = re.compile("|".join(re.escape(token) for token in _LEGACY_REMAP))

_WEIGHT_KEYS = {
    "rouge1_weight": "rouge1",
    "rouge2_weight": "rouge2",
    "rougeL_weight": "rougeL",
}


def _text(value: Any) -> str:
    return str(value or "")


def _legacy_prediction_text(text: str) -> str:
    """Mirror the legacy evaluator's prediction preamble/line handling."""
    kept_lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and _PREAMBLE not in line:
            kept_lines.append(line)
    return " ".join(kept_lines)


def _tokens(text: str, *, prediction: bool) -> list[str]:
    if prediction:
        text = _legacy_prediction_text(text)
    # Match the legacy Python preprocessor's case folding and bracket remap.
    # split() preserves Unicode token contents rather than using [a-z0-9].
    text = _LEGACY_REMAP_RE.sub(lambda match: _LEGACY_REMAP[match.group()], text.lower())
    return text.split()


def _f1(overlap: int, prediction_count: int, reference_count: int) -> float:
    if overlap <= 0 or prediction_count <= 0 or reference_count <= 0:
        return 0.0
    # This is 2 * precision * recall / (precision + recall), simplified.
    return 2.0 * overlap / (prediction_count + reference_count)


def _ngram_f1(prediction: list[str], reference: list[str], n: int) -> float:
    prediction_ngrams = Counter(zip(*(prediction[offset:] for offset in range(n))))
    reference_ngrams = Counter(zip(*(reference[offset:] for offset in range(n))))
    overlap = sum((prediction_ngrams & reference_ngrams).values())
    return _f1(overlap, sum(prediction_ngrams.values()), sum(reference_ngrams.values()))


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Return LCS length in O(len(left) * len(right)) time and O(min(n,m)) space."""
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


class RougeReward:
    """Compute weighted ROUGE-1/2/L F1 for one candidate and one reference.

    ``config`` accepts only ``rouge1_weight``, ``rouge2_weight``, and
    ``rougeL_weight``. Missing weights default to 1.0. The returned reward is
    their normalized weighted mean; all four returned values are in [0, 1].

    Tokens are whitespace-delimited and Unicode-preserving. Matching applies
    lowercase and the legacy evaluator's bracket remapping, but no stemming.
    Reasoning/wrapper tags are retained as ordinary text so the reward does
    not silently erase undesirable model behavior.
    """

    def __init__(self, config: Mapping[str, Any]):
        if not isinstance(config, Mapping):
            raise TypeError("reward config must be a mapping")
        unknown = set(config) - set(_WEIGHT_KEYS)
        if unknown:
            raise ValueError(f"Unknown reward config keys: {sorted(unknown)}")

        weights: dict[str, float] = {}
        for key, metric in _WEIGHT_KEYS.items():
            value = config.get(key, 1.0)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{key} must be a finite nonnegative number")
            weight = float(value)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
            weights[metric] = weight

        try:
            total_weight = math.fsum(weights.values())
        except OverflowError as exc:
            raise ValueError("reward weights must have a finite positive sum") from exc
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("at least one reward weight must be positive")
        self._weights = {metric: weight / total_weight for metric, weight in weights.items()}

    def __call__(self, prediction: str, reference: str) -> dict[str, float]:
        prediction_tokens = _tokens(_text(prediction), prediction=True)
        reference_tokens = _tokens(_text(reference), prediction=False)

        scores = {
            "rouge1": _ngram_f1(prediction_tokens, reference_tokens, 1),
            "rouge2": _ngram_f1(prediction_tokens, reference_tokens, 2),
            "rougeL": _f1(
                _lcs_length(prediction_tokens, reference_tokens),
                len(prediction_tokens),
                len(reference_tokens),
            ),
        }
        reward = math.fsum(self._weights[metric] * scores[metric] for metric in scores)
        return {
            "reward": min(1.0, max(0.0, float(reward))),
            **{metric: min(1.0, max(0.0, float(score))) for metric, score in scores.items()},
        }
