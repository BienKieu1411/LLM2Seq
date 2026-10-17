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
        self.max_band_cost = float(config.pop("max_band_cost", 2.0))
        if not math.isfinite(self.max_band_cost) or self.max_band_cost <= 0:
            raise ValueError("max_band_cost must be a finite positive number")
        self.band_deadzone = float(config.pop("band_deadzone", 0.0))
        if not math.isfinite(self.band_deadzone) or self.band_deadzone < 0:
            raise ValueError("band_deadzone must be a finite nonnegative number")
        self.reward_mode = str(config.pop("reward_mode", "linear"))
        if self.reward_mode not in {"linear", "gr3"}:
            raise ValueError("reward_mode must be 'linear' or 'gr3'")
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
        effective_error = max(0.0, error - self.band_deadzone * error_budget)
        cost = min(self.max_band_cost, effective_error / error_budget)
        if self.reward_mode == "gr3":
            reward = quality * math.exp(-self.length_weight * cost) + source_weight * source_score
        else:
            reward = quality + source_weight * source_score - self.length_weight * cost
        return {
            **scores,
            "quality": quality,
            "reward": reward,
            "generated_words": actual,
            "target_words": target_words,
            "abs_len_loss": error,
            "length_cost": cost,
            "length_deviation": (actual - target_words) / target_words,
            "source_score": source_score,
        }
