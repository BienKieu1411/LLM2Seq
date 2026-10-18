import math
import random

import pytest
import torch
from seam_grpo.config import cosine_multiplier
from seam_grpo.objective import clipped_terms, group_advantages
from seam_grpo.reward import RougeReward, _lcs_length


def test_zero_valued_centered_loss_still_has_policy_gradient():
    rewards = torch.tensor([0.8, 0.2, 0.5, 0.5])
    advantages = group_advantages(rewards, 2)
    torch.testing.assert_close(advantages, torch.tensor([0.3, -0.3, 0.0, 0.0]))
    logps = torch.full((4, 2), -2.0, requires_grad=True)
    mask = torch.tensor([[True, True], [True, False], [True, True], [True, True]])
    loss, _ = clipped_terms(logps, logps.detach().clone(), advantages, mask)
    loss.backward()
    assert logps.grad[0, 0] < 0
    assert logps.grad[1, 0] > 0
    assert logps.grad[1, 1] == 0
    assert not logps.grad[2:].any()


def test_clipping_sides_and_padding():
    current = torch.tensor([[math.log(1.5)], [math.log(0.5)]], requires_grad=True)
    loss, _ = clipped_terms(
        current, torch.zeros_like(current), torch.tensor([1.0, -1.0]), torch.ones_like(current, dtype=torch.bool)
    )
    loss.backward()
    assert not current.grad.any()
    dirty = torch.tensor([[-2.0, float("nan")]], requires_grad=True)
    loss, _ = clipped_terms(
        dirty, torch.tensor([[-2.0, float("nan")]]), torch.tensor([1.0]), torch.tensor([[True, False]])
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert dirty.grad[0, 1] == 0


def test_kl_and_group_ties():
    assert not group_advantages(torch.ones(4), 2, "group").any()
    logps = torch.tensor([[-2.0]], requires_grad=True)
    loss, metrics = clipped_terms(
        logps,
        logps.detach(),
        torch.zeros(1),
        torch.ones(1, 1, dtype=torch.bool),
        ref_logps=torch.tensor([[-1.0]]),
        beta=0.01,
    )
    assert metrics["kl_sum"] > 0
    loss.backward()
    assert logps.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        group_advantages(torch.ones(3), 2)


def test_fixed_horizon_accumulation_has_identical_gradient():
    torch.manual_seed(9)
    current = torch.randn(4, 5, requires_grad=True)
    old = current.detach().clone()
    adv = torch.tensor([1.0, -1.0, 2.0, -2.0])
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    full, _ = clipped_terms(current, old, adv, mask)
    (full_grad,) = torch.autograd.grad(full / (4 * 5), current)
    split = current.detach().clone().requires_grad_()
    for start in range(0, 4, 2):
        loss, _ = clipped_terms(
            split[start : start + 2], old[start : start + 2], adv[start : start + 2], mask[start : start + 2]
        )
        (loss / (4 * 5)).backward()
    torch.testing.assert_close(full_grad, split.grad)


def test_unicode_reward_and_weighting():
    scorer = RougeReward({})
    assert scorer("Tóm tắt tiếng Việt có dấu.", "Tóm tắt tiếng Việt có dấu.")["reward"] == 1
    assert scorer("cá", "ca")["reward"] == 0
    result = scorer("a b x", "a b c")
    assert result["rouge1"] == pytest.approx(2 / 3)
    assert result["rouge2"] == pytest.approx(1 / 2)
    assert result["rougeL"] == pytest.approx(2 / 3)
    assert RougeReward({"rouge1_weight": 0, "rouge2_weight": 1, "rougeL_weight": 0})("a b x", "a b c")["reward"] == 0.5
    for cfg in (
        {"rouge2_weight": -1},
        {"rouge1_weight": float("inf")},
        {"bad": 1},
        {"rouge1_weight": 0, "rouge2_weight": 0, "rougeL_weight": 0},
    ):
        with pytest.raises((ValueError, TypeError)):
            RougeReward(cfg)


def test_cosine_decay_and_zero_warmup():
    assert cosine_multiplier(0, 100, 0, 0.1) == 1
    assert cosine_multiplier(100, 100, 0, 0.1) == pytest.approx(0.1)
    values = [cosine_multiplier(i, 100, 0, 0.1) for i in range(101)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_bitset_lcs_matches_original_dp_with_unicode_and_repetition():
    def reference(left, right):
        row = [0] * (len(right) + 1)
        for token in left:
            current = [0]
            for j, other in enumerate(right):
                current.append(row[j] + 1 if token == other else max(row[j + 1], current[-1]))
            row = current
        return row[-1]

    rng = random.Random(73)
    vocabulary = ["Việt", "việt", "cá", "ca", "tóm", "tắt", "1.2%", ","]
    for _ in range(80):
        left = rng.choices(vocabulary, k=rng.randrange(150))
        right = rng.choices(vocabulary, k=rng.randrange(150))
        assert _lcs_length(left, right) == reference(left, right)
    assert _lcs_length([], []) == 0
    assert _lcs_length(["x"] * 2048, ["x"] * 1024) == 1024
