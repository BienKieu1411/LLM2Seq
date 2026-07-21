import torch

from evibridge.bridge import EvidenceBridge


def _inputs():
    torch.manual_seed(1)
    hidden = torch.randn(2, 9, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0]])
    unit_ids = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 0], [0, 1, 1, 2, 2, 0, 0, 0, 0]])
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
    return hidden, mask, unit_ids, labels


def test_evidence_bridge_produces_dual_memory_and_auxiliary_gradients():
    bridge = EvidenceBridge(
        32,
        32,
        {
            "mode": "evidence",
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_size": 64,
            "num_evidence_slots": 4,
            "dropout": 0.0,
        },
    )
    hidden, mask, unit_ids, labels = _inputs()
    output = bridge(hidden, mask, unit_ids, labels)
    assert output.memory.shape == (2, 13, 32)
    assert output.memory_mask.shape == (2, 13)
    assert output.salience_logits.shape == (2, 3)
    assert torch.isfinite(output.loss_evidence)
    (output.memory.mean() + output.loss_evidence + output.loss_diversity).backward()
    assert bridge.salience_head[-1].weight.grad is not None
    assert bridge.slot_queries.grad is not None


def test_lamate_style_bridge_is_a_token_level_bidirectional_baseline():
    bridge = EvidenceBridge(
        32,
        48,
        {
            "mode": "lamate",
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 4,
            "ffn_size": 64,
            "dropout": 0.0,
        },
    )
    hidden, mask, unit_ids, labels = _inputs()
    output = bridge(hidden, mask, unit_ids, labels)
    assert output.memory.shape == (2, 9, 48)
    assert torch.equal(output.memory_mask, mask)
    assert output.salience_logits is None


def test_slots_only_ablation_removes_full_token_memory():
    bridge = EvidenceBridge(
        32,
        32,
        {
            "mode": "slots_only",
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 4,
            "ffn_size": 64,
            "num_evidence_slots": 3,
            "dropout": 0.0,
        },
    )
    output = bridge(*_inputs())
    assert output.memory.shape == (2, 3, 32)
    assert output.memory_mask.tolist() == [[1, 1, 1], [1, 1, 1]]
