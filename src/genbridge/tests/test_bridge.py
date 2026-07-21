import torch
from genbridge.bridge import SummaryBridge


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
    assert output.salience_logits.shape == (2, 3)
    assert torch.isfinite(output.loss_salience)
    (output.memory.mean() + output.loss_salience + output.loss_plan_diversity).backward()
    assert bridge.salience_head[-1].weight.grad is not None
    assert bridge.plan_attention.in_proj_weight.grad is not None


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
