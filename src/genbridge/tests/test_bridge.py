import torch
from genbridge.bridge import (
    BidirectionalRotaryEncoder,
    SummaryBridge,
    _balanced_binary_cross_entropy,
    _ordered_plan_evidence_loss,
)


def _inputs():
    torch.manual_seed(1)
    hidden = torch.randn(2, 9, 32)
    plans = torch.randn(2, 4, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0]])
    unit_ids = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 0], [0, 1, 1, 2, 2, 0, 0, 0, 0]])
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
    return hidden, plans, mask, unit_ids, labels


def test_evidence_bridge_produces_dual_memory_and_auxiliary_gradients():
    bridge = SummaryBridge(
        32,
        32,
        {
            "mode": "genbridge",
            "hidden_size": 32,
            "token_num_layers": 2,
            "unit_num_layers": 1,
            "num_heads": 4,
            "ffn_size": 64,
            "dropout": 0.0,
        },
    )
    hidden, plans, mask, unit_ids, labels = _inputs()
    output = bridge(hidden, plans, mask, unit_ids, labels)
    assert output.memory.shape == (2, 13, 32)
    assert output.memory_mask.shape == (2, 13)
    assert output.token_memory.shape == (2, 9, 32)
    assert torch.equal(output.token_memory_mask, mask)
    assert output.plan_memory.shape == (2, 4, 32)
    assert output.plan_memory_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]
    assert output.token_attention_bias.shape == (2, 9)
    assert torch.count_nonzero(output.token_attention_bias[unit_ids.eq(0)]) == 0
    assert output.salience_logits.shape == (2, 3)
    assert isinstance(bridge.token_skip, torch.nn.Identity)
    assert isinstance(bridge.plan_skip, torch.nn.Identity)
    assert isinstance(bridge.token_output_norm, torch.nn.Identity)
    assert isinstance(bridge.plan_output_norm, torch.nn.Identity)
    assert abs(torch.tanh(bridge.token_adapter_gate).item() - 0.1) < 1e-6
    assert abs(torch.tanh(bridge.plan_adapter_gate).item() - 0.1) < 1e-6
    assert torch.isfinite(output.loss_salience)
    assert torch.isfinite(output.loss_plan_evidence)
    (
        output.memory.mean()
        + output.token_attention_bias.mean()
        + output.loss_salience
        + output.loss_plan_evidence
        + output.loss_plan_diversity
    ).backward()
    assert bridge.salience_head[-1].weight.grad is not None
    assert bridge.plan_attention.in_proj_weight.grad is not None
    assert bridge.token_adapter_gate.grad is not None
    assert bridge.plan_adapter_gate.grad is not None


def test_ordered_plan_evidence_alignment_rewards_source_order():
    units = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    labels = torch.tensor([[1.0, 0.0, 1.0]])
    aligned = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    reversed_plans = aligned.flip(1)
    aligned_loss = _ordered_plan_evidence_loss(aligned, units, mask, labels, temperature=0.1)
    reversed_loss = _ordered_plan_evidence_loss(reversed_plans, units, mask, labels, temperature=0.1)
    assert aligned_loss < reversed_loss


def test_balanced_salience_loss_does_not_dilute_the_positive_class():
    positive = torch.tensor([-2.0])
    one_negative = torch.tensor([-2.0])
    many_negatives = one_negative.repeat(20)
    first_logits = torch.cat([positive, one_negative])
    second_logits = torch.cat([positive, many_negatives])
    first_labels = torch.tensor([1.0, 0.0])
    second_labels = torch.tensor([1.0] + [0.0] * 20)
    first = _balanced_binary_cross_entropy(
        first_logits,
        first_labels,
        torch.ones_like(first_labels, dtype=torch.bool),
    )
    second = _balanced_binary_cross_entropy(
        second_logits,
        second_labels,
        torch.ones_like(second_labels, dtype=torch.bool),
    )
    torch.testing.assert_close(first, second)


def test_rotary_bidirectional_adapter_is_left_padding_invariant():
    torch.manual_seed(7)
    encoder = BidirectionalRotaryEncoder(
        hidden_size=16,
        num_heads=4,
        ffn_size=32,
        dropout=0.0,
        num_layers=2,
        rope_theta=1_000_000.0,
        use_rope=True,
    ).eval()
    assert not torch.equal(
        encoder.layers[0].attention.qkv_proj.weight,
        encoder.layers[1].attention.qkv_proj.weight,
    )
    real = torch.randn(1, 5, 16)
    unpadded = encoder(real, torch.ones(1, 5, dtype=torch.long))
    left_padded = encoder(
        torch.cat([torch.randn(1, 3, 16), real], dim=1),
        torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]]),
    )
    torch.testing.assert_close(unpadded, left_padded[:, -5:], atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(left_padded[:, :3]) == 0


def test_no_rope_ablation_changes_only_position_operation_not_parameters():
    torch.manual_seed(11)
    rotary = BidirectionalRotaryEncoder(16, 4, 32, 0.0, 1, 1_000_000.0, True).eval()
    without = BidirectionalRotaryEncoder(16, 4, 32, 0.0, 1, 1_000_000.0, False).eval()
    without.load_state_dict(rotary.state_dict(), strict=True)
    assert sum(parameter.numel() for parameter in rotary.parameters()) == sum(
        parameter.numel() for parameter in without.parameters()
    )
    hidden = torch.randn(2, 6, 16)
    mask = torch.ones(2, 6, dtype=torch.long)
    assert not torch.allclose(rotary(hidden, mask), without(hidden, mask))


def test_lamate_style_bridge_is_a_token_level_bidirectional_baseline():
    bridge = SummaryBridge(
        32,
        48,
        {
            "mode": "lamate",
            "hidden_size": 32,
            "token_num_layers": 1,
            "num_heads": 4,
            "ffn_size": 64,
            "dropout": 0.0,
        },
    )
    hidden, plans, mask, unit_ids, labels = _inputs()
    output = bridge(hidden, plans, mask, unit_ids, labels)
    assert output.memory.shape == (2, 9, 48)
    assert torch.equal(output.memory_mask, mask)
    assert output.token_memory.shape == (2, 9, 48)
    assert output.salience_logits is None


def test_plan_only_ablation_removes_full_token_memory():
    bridge = SummaryBridge(
        32,
        32,
        {
            "mode": "plan_only",
            "hidden_size": 32,
            "token_num_layers": 1,
            "unit_num_layers": 1,
            "num_heads": 4,
            "ffn_size": 64,
            "dropout": 0.0,
        },
    )
    output = bridge(*_inputs())
    assert output.memory.shape == (2, 4, 32)
    assert output.memory_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]
