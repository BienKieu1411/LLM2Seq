import torch
from adabimask.mask_policy import (
    LayerMaskPolicy,
    MaskPolicyConfig,
    balanced_layer_groups,
    choose_fixed_groups,
)


def test_qwen35_24_layers_follow_six_native_groups_of_four():
    groups = balanced_layer_groups(24, 6)
    assert groups == [tuple(range(start, start + 4)) for start in range(0, 24, 4)]


def test_fixed_ablation_strategies_select_equal_budgets():
    assert choose_fixed_groups(MaskPolicyConfig(num_groups=6, budget_groups=2, fixed_strategy="bottom")) == (0, 1)
    assert choose_fixed_groups(MaskPolicyConfig(num_groups=6, budget_groups=2, fixed_strategy="middle")) == (2, 3)
    assert choose_fixed_groups(MaskPolicyConfig(num_groups=6, budget_groups=2, fixed_strategy="top")) == (4, 5)
    random_selection = choose_fixed_groups(
        MaskPolicyConfig(num_groups=6, budget_groups=2, fixed_strategy="random", random_seed=42)
    )
    assert len(random_selection) == 2
    assert random_selection == choose_fixed_groups(
        MaskPolicyConfig(num_groups=6, budget_groups=2, fixed_strategy="random", random_seed=42)
    )


def test_learnable_policy_uses_soft_routes_in_train_and_hard_topk_in_eval():
    policy = LayerMaskPolicy(
        24,
        MaskPolicyConfig(mode="learnable", num_groups=6, budget_groups=2, init_probability=0.25, hard_eval=True),
    )
    policy.train()
    assert all(policy.route(layer) == "mix" for layer in range(24))

    with torch.no_grad():
        policy.gate_logits.copy_(torch.tensor([-3.0, 4.0, 1.0, -2.0, 3.0, 0.0]))
    policy.eval()
    assert policy.topk_groups() == (1, 4)
    assert policy.selected_layers() == tuple(range(4, 8)) + tuple(range(16, 20))
    assert policy.route(5) == "bidirectional"
    assert policy.route(12) == "causal"


def test_gate_regularization_backpropagates():
    policy = LayerMaskPolicy(
        24,
        MaskPolicyConfig(mode="learnable", num_groups=6, budget_groups=2, budget_weight=0.5, binary_weight=0.1),
    )
    losses = policy.regularization()
    losses["loss_gate"].backward()
    assert policy.gate_logits.grad is not None
    assert torch.isfinite(policy.gate_logits.grad).all()
