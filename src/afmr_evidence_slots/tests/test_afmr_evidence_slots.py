import copy
from pathlib import Path

import pytest
import torch
from eviseq_afmr.config import evidence_slot_settings, load_config, validate_config
from eviseq_afmr.modeling.afmr import AdaptiveFullMemoryResidualBridge
from eviseq_afmr.modeling.evidence_slots import EvidenceSlotBridge
from eviseq_afmr.modeling.outputs import EncoderState


def _slot_config(prior_enabled: bool = False) -> dict:
    return {
        "enabled": True,
        "num_slots": 4,
        "dim": 16,
        "num_heads": 4,
        "rank": 8,
        "refine_layers": 1,
        "gate_init": 0.05,
        "gate_max": 0.30,
        "prior_enabled": prior_enabled,
        "prior_init": 0.05,
        "prior_max": 0.30,
    }


def _architecture(*, direct: bool = False, prior_enabled: bool = False) -> dict:
    return {
        "name": "afmr_value_anchor",
        "bridge_mode": "direct_projection" if direct else "afmr",
        "controller_dim": 8,
        "depth_taps": 2,
        "depth_rank": 4,
        "depth_gate_init": 0.02,
        "depth_gate_max": 0.15,
        "feature_rank": 4,
        "feature_gate_init": 0.02,
        "feature_gate_max": 0.20,
        "focus_hidden": 8,
        "focus_windows": [2, 4],
        "focus_overlap": 0.5,
        "focus_strength_init": 0.10,
        "focus_strength_max": 1.0,
        "temperature_init": 1.0,
        "temperature_min": 0.5,
        "temperature_max": 2.0,
        "contextual_value": {"enabled": False},
        "evidence_slots": _slot_config(prior_enabled),
    }


def test_competitive_read_normalizes_each_slot_and_handles_empty_rows():
    logits = torch.randn(2, 3, 4, 7)
    mask = torch.tensor(
        [[False, True, True, False, True, True, False], [False, False, False, False, False, False, False]]
    )
    weights = EvidenceSlotBridge._competitive_read_weights(logits, mask)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights[0].sum(dim=-1), torch.ones(3, 4))
    torch.testing.assert_close(weights[1], torch.zeros_like(weights[1]))
    assert weights[..., ~mask[0]][0].eq(0).all()


def test_initial_slots_keep_orthogonal_identity_before_controller_learning():
    torch.manual_seed(2)
    module = EvidenceSlotBridge(12, 12, 8, _slot_config())
    slots = module.slot_embeddings.detach()
    cosine = torch.nn.functional.normalize(slots, dim=-1) @ torch.nn.functional.normalize(slots, dim=-1).T
    torch.testing.assert_close(cosine, torch.eye(module.num_slots), atol=1.0e-6, rtol=0)
    torch.testing.assert_close(
        slots.square().mean(dim=-1).sqrt(),
        torch.ones(module.num_slots),
        atol=1.0e-6,
        rtol=0,
    )
    assert module.controller_to_slots.weight.count_nonzero() == 0
    projected = module.read_query(slots)
    assert torch.linalg.matrix_rank(projected) == module.num_slots


@pytest.mark.parametrize("prior_enabled", [False, True])
def test_masks_prefix_padding_and_all_invalid_rows(prior_enabled):
    torch.manual_seed(3)
    module = EvidenceSlotBridge(12, 12, 8, _slot_config(prior_enabled))
    source = torch.randn(2, 7, 12)
    anchor = torch.randn(2, 7, 12)
    controller = torch.randn(2, 8)
    content = torch.tensor(
        [[False, False, True, True, True, False, False], [False, False, False, False, False, False, False]]
    )
    residual, prior = module(source, content, controller, anchor)
    assert torch.isfinite(residual).all() and torch.isfinite(prior).all()
    assert residual.masked_select(~content[..., None]).eq(0).all()
    assert prior.masked_select(~content).eq(0).all()
    assert residual[0, content[0]].abs().sum() > 0
    torch.testing.assert_close(residual[1], torch.zeros_like(residual[1]))
    torch.testing.assert_close(prior[1], torch.zeros_like(prior[1]))
    if prior_enabled:
        assert prior[0, content[0]].abs().sum() > 0
        assert prior[0, content[0]].abs().max() <= module.prior_max + 1.0e-6
    else:
        torch.testing.assert_close(prior, torch.zeros_like(prior))


def test_first_backward_reaches_slot_read_mix_topdown_key_and_optional_prior():
    torch.manual_seed(7)
    module = EvidenceSlotBridge(12, 12, 8, _slot_config(prior_enabled=True))
    source = torch.randn(2, 9, 12, requires_grad=True)
    anchor = torch.randn(2, 9, 12, requires_grad=True)
    controller = torch.randn(2, 8, requires_grad=True)
    content = torch.ones(2, 9, dtype=torch.bool)
    content[:, :2] = False
    residual, prior = module(source, content, controller, anchor)
    residual_weight = torch.randn_like(residual)
    prior_weight = torch.randn_like(prior)
    (residual.mul(residual_weight).sum() + prior.mul(prior_weight).sum()).backward()
    required = (
        "slot_embeddings",
        "controller_to_slots.weight",
        "read_query.weight",
        "read_key.weight",
        "read_value.weight",
        "read_output.weight",
        "refinement.0.attention.in_proj_weight",
        "refinement.0.mlp.0.weight",
        "topdown_query.weight",
        "topdown_key.weight",
        "topdown_value.weight",
        "topdown_output.weight",
        "key_down.weight",
        "key_up.weight",
        "gate_raw.weight",
        "prior_source.weight",
        "prior_context.weight",
        "prior_output.weight",
        "prior_raw.weight",
    )
    parameters = dict(module.named_parameters())
    for name in required:
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name


def test_evidence_route_is_deterministic_between_train_and_eval():
    torch.manual_seed(11)
    module = EvidenceSlotBridge(12, 12, 8, _slot_config(prior_enabled=True))
    arguments = (
        torch.randn(2, 8, 12),
        torch.tensor([[False, True, True, True, True, True, False, False]] * 2),
        torch.randn(2, 8),
        torch.randn(2, 8, 12),
    )
    module.train()
    train_outputs = module(*arguments)
    module.eval()
    eval_outputs = module(*arguments)
    for train_value, eval_value in zip(train_outputs, eval_outputs):
        torch.testing.assert_close(train_value, eval_value, rtol=0, atol=0)


def test_tiny_nonzero_initialization_is_bounded_by_controller_gate():
    torch.manual_seed(12)
    module = EvidenceSlotBridge(12, 12, 8, _slot_config())
    source = torch.randn(2, 8, 12)
    anchor = torch.randn(2, 8, 12)
    content = torch.ones(2, 8, dtype=torch.bool)
    residual, _ = module(source, content, torch.randn(2, 8), anchor)
    residual_rms = residual.float().square().mean(-1).sqrt()
    anchor_rms = anchor.float().square().mean(-1).sqrt()
    assert residual.abs().sum() > 0
    assert torch.all(residual_rms <= 0.05 * anchor_rms + 1.0e-6)


def test_full_bridge_changes_only_key_memory_and_preserves_h0_value_anchor():
    torch.manual_seed(13)
    bridge = AdaptiveFullMemoryResidualBridge(12, 12, _architecture())
    final = torch.randn(2, 8, 12)
    attention = torch.ones(2, 8, dtype=torch.bool)
    content = attention.clone()
    content[:, :2] = False
    state = EncoderState(final, (final + 0.1, final), attention, content)
    output = bridge(state, torch.randn(2, 3, 12), torch.ones(2, 3, dtype=torch.bool), torch.tensor([32.0, 48.0]))
    torch.testing.assert_close(output.value_memory, final, rtol=0, atol=0)
    assert not torch.equal(output.memory, output.value_memory)
    assert bridge.evidence_slots is not None


def test_direct_projection_control_ignores_evidence_slot_configuration_exactly():
    final = torch.randn(2, 8, 12)
    attention = torch.ones(2, 8, dtype=torch.bool)
    content = attention.clone()
    content[:, :2] = False
    state = EncoderState(final, (final, final), attention, content)
    prompt = torch.randn(2, 3, 12)
    prompt_mask = torch.ones(2, 3, dtype=torch.bool)
    budget = torch.tensor([32.0, 48.0])
    enabled = _architecture(direct=True)
    disabled = copy.deepcopy(enabled)
    disabled["evidence_slots"]["enabled"] = False
    torch.manual_seed(17)
    bridge_enabled = AdaptiveFullMemoryResidualBridge(12, 12, enabled)
    torch.manual_seed(17)
    bridge_disabled = AdaptiveFullMemoryResidualBridge(12, 12, disabled)
    left = bridge_enabled(state, prompt, prompt_mask, budget)
    right = bridge_disabled(state, prompt, prompt_mask, budget)
    assert not any("evidence_slots" in name for name, _ in bridge_enabled.named_parameters())
    for left_value, right_value in zip(left.__dict__.values(), right.__dict__.values()):
        if isinstance(left_value, torch.Tensor):
            torch.testing.assert_close(left_value, right_value, rtol=0, atol=0)


def test_nested_config_validation_and_pubmed_defaults():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_pubmed.yaml")
    slots = evidence_slot_settings(config["architecture"])
    assert slots["enabled"] is True
    assert slots["num_slots"] == 32
    assert slots["dim"] == slots["rank"] == 256
    assert slots["num_heads"] == 4 and slots["refine_layers"] == 1
    assert slots["gate_init"] == 0.05 and slots["gate_max"] == 0.30
    assert slots["prior_enabled"] is False
    assert config["training"]["salience_loss_weight"] == 0.0
    assert "evidence_slots" in config["experiment"]["output_dir"]

    invalid = copy.deepcopy(config)
    invalid["architecture"]["evidence_slots"]["dim"] = 255
    with pytest.raises(ValueError, match="divisible"):
        validate_config(invalid)
