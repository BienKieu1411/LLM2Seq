"""Output-based rewards; word counting matches the Vietnamese length evaluator."""

from __future__ import annotations

import math
import re
import unicodedata

from seam_grpo.reward import RougeReward

WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", flags=re.UNICODE)


def count_words(text: str) -> int:
    return len(WORD_RE.findall(unicodedata.normalize("NFKC", text)))


class LengthReward:
    def __init__(self, config):
        config = dict(config)
        self.length_weight = config.pop("length_weight", 0.1)
        if isinstance(self.length_weight, bool) or not isinstance(self.length_weight, (float, int)):
            raise ValueError("length_weight must be a finite nonnegative number")
        if not math.isfinite(self.length_weight) or self.length_weight < 0:
            raise ValueError("length_weight must be a finite nonnegative number")
        self.rouge = RougeReward(config)

    def __call__(self, prediction, reference, target_words, error_budget, source_score=0.0, source_weight=0.0):
        if isinstance(target_words, bool) or not isinstance(target_words, int) or target_words < 1:
            raise ValueError("target_words must be a positive integer")
        if not math.isfinite(error_budget) or error_budget <= 0:
            raise ValueError("error_budget must be positive")
        if not math.isfinite(source_score) or not -1 <= source_score <= 1:
            raise ValueError("source_score must be finite in [-1,1]")
        scores = self.rouge(prediction, reference)
        quality = scores.pop("reward")
        actual = count_words(prediction)
        error = abs(actual - target_words)
        cost = error / error_budget
        return {
            **scores,
            "quality": quality,
            "reward": quality + source_weight * source_score - self.length_weight * cost,
            "generated_words": actual,
            "target_words": target_words,
            "abs_len_loss": error,
            "length_cost": cost,
            "length_deviation": (actual - target_words) / target_words,
            "source_score": source_score,
        }
