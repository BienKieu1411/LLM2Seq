import pytest
import yaml
from fixtures import settings_file
from seam_length_grpo.config import load
from seam_length_grpo.reward import LengthReward


def test_source_reward_is_enabled_when_not_explicitly_overridden(tmp_path):
    _, _, path = settings_file(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.pop("source_reward")
    path.write_text(yaml.safe_dump(raw))
    settings = load(path)
    assert settings["source_reward"]["weight"] == 0.05
    assert settings["source_reward"]["teacher_checkpoint"] == settings["sft_checkpoint"]


def test_three_term_reward_uses_per_sample_word_request():
    reward = LengthReward({"rouge1_weight": 0.25, "rouge2_weight": 0.5, "rougeL_weight": 0.25, "length_weight": 0.1})
    prediction, reference = "one two three four", "one two three four"
    short = reward(prediction, reference, target_words=6, error_budget=2, source_score=0.6, source_weight=0.05)
    assert short["reward"] == pytest.approx(short["quality"] - 0.1 + 0.03)
    exact = reward(prediction, reference, target_words=4, error_budget=2, source_score=0.6, source_weight=0.05)
    assert exact["reward"] == pytest.approx(exact["quality"] + 0.03)
    assert exact["reward"] - short["reward"] == pytest.approx(0.1)
