"""Pure-tensor tests for the V5 prospective-summary bridge."""

from __future__ import annotations

import torch
from llm2seq_v5.adapter import (
    IdentityResidualProjection,
    ProspectiveSummaryPlanner,
    StableTokenLayerFusion,
    SummaryAdapterV2,
)
from llm2seq_v5.response_alignment import ordered_response_alignment_loss
from llm2seq_v5.training import _capture_optimizer_moments, _restore_optimizer_moments


def test_raw_final_layer_highway_is_exact_at_zero_gate():
    torch.manual_seed(3)
    final = torch.randn(2, 5, 8)
    hidden_states = (torch.randn_like(final), torch.randn_like(final), final)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    fusion = StableTokenLayerFusion(8, [-1, -2, -3], dropout=0.0, gate_init=0.0)

    actual = fusion(hidden_states, mask)
    expected = final.masked_fill(~mask.bool().unsqueeze(-1), 0)
    assert torch.equal(actual, expected)

    # Earlier depths may change arbitrarily without perturbing the highway.
    changed = (hidden_states[0] * 1000, hidden_states[1] - 500, final)
    assert torch.equal(fusion(changed, mask), expected)


def test_same_width_projection_is_exact_identity_at_zero_gate():
    torch.manual_seed(5)
    states = torch.randn(3, 4, 8)
    projection = IdentityResidualProjection(8, 8, 16, dropout=0.0, gate_init=0.0)
    assert torch.equal(projection(states), states)


def test_planner_shape_masking_and_gradients():
    torch.manual_seed(7)
    planner = ProspectiveSummaryPlanner(
        hidden_size=8,
        num_slots=4,
        num_layers=2,
        num_heads=2,
        ffn_size=24,
        dropout=0.0,
        gate_init=0.2,
    ).eval()
    memory = torch.randn(2, 6, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
    salience_bias = torch.tensor([[0.4, -0.2, 0.1, 0.0, 0.0, 0.0], [0.1, 0.2, -0.3, 0.5, 0.0, 0.0]])

    output = planner(memory, mask, salience_bias)
    assert output.shape == (2, 4, 8)

    perturbed = memory.detach().clone()
    perturbed[~mask.bool()] = 100_000.0
    with torch.no_grad():
        masked_output = planner(perturbed, mask, salience_bias)
    assert torch.allclose(output.detach(), masked_output, atol=1e-5, rtol=1e-5)

    output.square().mean().backward()
    assert planner.ordered_slots.grad is not None
    assert planner.ordered_slots.grad.abs().sum() > 0
    assert memory.grad is not None
    assert memory.grad[mask.bool()].abs().sum() > 0
    assert torch.equal(memory.grad[~mask.bool()], torch.zeros_like(memory.grad[~mask.bool()]))


def _tiny_salience_adapter() -> SummaryAdapterV2:
    adapter = SummaryAdapterV2(
        8,
        8,
        {
            "hidden_size": 8,
            "layer_fusion": False,
            "projection_ffn_size": 16,
            "projection_gate_init": 0.0,
            "num_bidirectional_layers": 0,
            "num_heads": 2,
            "dropout": 0.0,
            "use_salience": True,
            "salience_hidden_size": 4,
            "salience_gate_init": 0.5,
            "salience_bias_scale": 1.0,
            "use_summary_planner": False,
        },
    )
    assert adapter.salience_head is not None
    with torch.no_grad():
        for parameter in adapter.salience_head.parameters():
            parameter.zero_()
    return adapter


def test_oracle_salience_is_train_only_and_default_is_prediction_only():
    states = (torch.randn(1, 4, 8),)
    mask = torch.ones(1, 4, dtype=torch.long)
    unit_ids = torch.tensor([[1, 1, 2, 2]])
    labels = torch.tensor([[1.0, 0.0]])
    adapter = _tiny_salience_adapter()

    adapter.train()
    predicted = adapter(states, mask, unit_ids, labels)
    assert adapter.oracle_evidence_mix == 0.0
    assert torch.allclose(predicted.attention_bias, torch.zeros_like(predicted.attention_bias))

    adapter.set_oracle_evidence_mix(1.0)
    oracle = adapter(states, mask, unit_ids, labels)
    assert torch.all(oracle.attention_bias[:, :2] > 0)
    assert torch.all(oracle.attention_bias[:, 2:] < 0)

    # Eval ignores both the schedule and labels: no target leakage is possible.
    adapter.eval()
    eval_with_labels = adapter(states, mask, unit_ids, labels)
    eval_without_labels = adapter(states, mask, unit_ids, None)
    assert torch.allclose(eval_with_labels.attention_bias, eval_without_labels.attention_bias)
    assert torch.allclose(
        eval_with_labels.attention_bias,
        torch.zeros_like(eval_with_labels.attention_bias),
    )


def test_adapter_emits_ordered_prefix_and_mask():
    torch.manual_seed(11)
    adapter = SummaryAdapterV2(
        8,
        8,
        {
            "hidden_size": 8,
            "layer_fusion": True,
            "fuse_layers": [-1, -2],
            "layer_fusion_gate_init": 0.0,
            "projection_gate_init": 0.0,
            "projection_ffn_size": 16,
            "num_bidirectional_layers": 0,
            "num_heads": 2,
            "dropout": 0.0,
            "use_salience": False,
            "use_summary_planner": True,
            "num_summary_slots": 4,
            "summary_planner_layers": 2,
            "summary_planner_heads": 2,
            "summary_planner_ffn_size": 24,
        },
    ).eval()
    hidden_states = (torch.randn(2, 5, 8), torch.randn(2, 5, 8))
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    output = adapter(hidden_states, mask)
    assert output.summary_prefix is not None
    assert output.summary_prefix.shape == (2, 4, 8)
    assert output.summary_prefix_mask is not None
    assert torch.equal(output.summary_prefix_mask, torch.ones(2, 4, dtype=mask.dtype))


def test_ordered_response_alignment_is_order_sensitive_and_slot_only():
    # Four orthogonal token embeddings make the expected slot order exact.
    embedding_weight = torch.eye(8, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 3, -100]])
    ordered = torch.eye(8)[:4].unsqueeze(0).clone().requires_grad_(True)
    reversed_slots = ordered.detach().flip(1).clone().requires_grad_(True)

    good = ordered_response_alignment_loss(ordered, labels, embedding_weight, temperature=0.1)
    bad = ordered_response_alignment_loss(reversed_slots, labels, embedding_weight, temperature=0.1)
    assert set(good) == {"loss", "cosine", "accuracy", "valid_slots"}
    assert good["valid_slots"].item() == 4
    assert good["cosine"].item() > 0.999
    assert good["accuracy"].item() == 1.0
    assert good["loss"] < bad["loss"]

    good["loss"].backward()
    assert ordered.grad is not None
    assert embedding_weight.grad is None


def test_alignment_handles_short_and_empty_targets():
    slots = torch.randn(2, 4, 6, requires_grad=True)
    labels = torch.tensor([[2, 3, -100], [-100, -100, -100]])
    embedding_weight = torch.randn(10, 6, requires_grad=True)
    result = ordered_response_alignment_loss(slots, labels, embedding_weight)
    assert result["valid_slots"].item() == 2
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert slots.grad is not None
    assert embedding_weight.grad is None


def test_vectorized_alignment_preserves_uneven_ordered_chunk_boundaries():
    # N=5, K=3 must retain the original contiguous split [0], [1,2], [3,4].
    embeddings = torch.eye(6)
    expected_targets = torch.stack(
        [
            embeddings[0],
            embeddings[[1, 2]].mean(dim=0),
            embeddings[[3, 4]].mean(dim=0),
        ]
    ).unsqueeze(0)
    slots = expected_targets.clone().requires_grad_(True)
    result = ordered_response_alignment_loss(
        slots,
        torch.tensor([[0, 1, 2, 3, 4, -100]]),
        embeddings,
        temperature=0.1,
    )
    assert result["valid_slots"].item() == 3
    assert result["cosine"].item() > 0.99999
    assert result["accuracy"].item() == 1.0


def test_interface_adam_moments_survive_the_stage_boundary():
    layer = torch.nn.Linear(4, 3)
    first = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    layer(torch.randn(2, 4)).square().mean().backward()
    first.step()
    carried = _capture_optimizer_moments(layer, first)
    assert carried

    expected = {name: state["exp_avg"].clone() for name, state in carried.items()}
    second = torch.optim.AdamW(layer.parameters(), lr=1e-4)
    restored = _restore_optimizer_moments(layer, second, carried)
    assert restored == len(expected)
    for name, parameter in layer.named_parameters():
        assert torch.equal(second.state[parameter]["exp_avg"], expected[name])
