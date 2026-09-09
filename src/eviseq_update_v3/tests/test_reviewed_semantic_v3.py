"""Regressions from the Luna review: short-source routing and real head suppression."""

import copy
import math

import pytest
import torch
from eviseq_update_v3.config import load_config, validate_config
from eviseq_update_v3.modeling.grounded_copy import GroundedCopyHead
from eviseq_update_v3.modeling.model import EviSeqAFMR
from eviseq_update_v3.modeling.semantic_plan import ReadPlan
from eviseq_update_v3.training.checkpoint import architecture_spec, load_checkpoint, save_checkpoint
from test_planned_semantic_v3 import config_planned, problem


def short_source(tokens=128, rank=512):
    torch.manual_seed(917)
    head = GroundedCopyHead(
        8,
        4,
        0.05,
        semantic_read=True,
        semantic_rank=rank,
        semantic_num_heads=4,
        semantic_attention="hierarchical_coverage",
        semantic_max_relative_rms=0.1,
        semantic_fusion="norm_preserving",
        semantic_head_gate_position="post_norm",
        semantic_planner={"region_size": 64, "partition_heads": False},
    )
    memory = torch.randn(1, tokens, 8, requires_grad=True)
    mask = torch.ones(1, tokens, dtype=torch.bool)
    indices = torch.arange(tokens)[None]
    state = head.prepare(
        memory,
        torch.zeros(1, tokens),
        mask,
        torch.nn.Embedding(32, 8),
        copy_token_ids=indices % 32,
        copy_token_mask=mask,
        copy_encoder_indices=indices,
        copy_token_indices=indices,
        copy_alignment_weights=torch.ones(1, tokens),
    )
    empty = torch.zeros(1, 3, state.region_keys.shape[1])
    return head, state, ReadPlan(empty, empty.clone())


@pytest.mark.parametrize("tokens", [1, 64, 128, 256])
def test_every_head_can_read_the_first_region_even_on_short_sources(tokens):
    head, state, plan = short_source(tokens)
    query = torch.zeros(1, 3, 512)
    before = head._hierarchical_context(query, state, plan)
    changed = copy.copy(state)
    changed.semantic_values = state.semantic_values.clone()
    changed.semantic_values[:, : min(tokens, 64)] += 2
    difference = (head._hierarchical_context(query, changed, plan) - before).reshape(1, 3, 4, 128)
    # The former hard partition left heads without a route to this region.
    assert (difference.abs().sum(-1) > 0).all()
    assert torch.isfinite(difference).all()


def test_coverage_and_continuity_change_region_choice_with_only_two_regions():
    head, state, plan = short_source()
    query = torch.zeros(1, 3, 512)
    state.semantic_values = torch.ones_like(state.semantic_values)
    state.semantic_values[:, 64:] = 3
    before = head._hierarchical_context(query, state, plan)
    used = plan.coverage.clone()
    used[..., 0] = 80
    after_coverage = head._hierarchical_context(query, state, ReadPlan(used, plan.recent))
    recent = plan.recent.clone()
    recent[..., 0] = 1
    after_continuity = head._hierarchical_context(query, state, ReadPlan(plan.coverage, recent))
    assert (after_coverage > before).all()  # Less mass on used region 0 (value 1).
    assert (after_continuity < before).all()  # More mass on recent region 0.


@pytest.mark.parametrize("reviewed", [False, True])
def test_shared_head_gain_reduction_attenuates_only_after_normalization(reviewed):
    head, _, hidden, _, _, _, state, _, plan = problem(reviewed=reviewed)
    with torch.no_grad():
        head.semantic_output.weight.normal_(std=0.001)  # Stay below the fusion cap.
        head.semantic_head_gate.weight.zero_()
        head.semantic_head_gate.bias.zero_()
        head.semantic_gate.weight.normal_(std=0.1)
        baseline, copy_before = head.read(hidden, state, plan)
        head.semantic_head_gate.bias.fill_(math.log(0.05 / 0.95))  # gain 0.1
        reduced, copy_after = head.read(hidden, state, plan)
        ratio = (reduced[0] - hidden[0]).norm() / (baseline[0] - hidden[0]).norm()
    if reviewed:
        assert float(ratio) == pytest.approx(0.1, abs=0.01)
    else:
        assert float(ratio) > 0.95  # Reproduce the normalization cancellation.
    for before, after in zip(copy_before, copy_after):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_post_norm_zero_initialization_and_empty_source_preserve_base_hidden():
    head, _, hidden, _, _, _, state, _, plan = problem(active=False, reviewed=True)
    generated, _ = head.read(hidden, state, plan)
    torch.testing.assert_close(generated, hidden, rtol=0, atol=0)
    with torch.no_grad():
        head.semantic_output.weight.normal_()
    generated, _ = head.read(hidden, state, plan)
    torch.testing.assert_close(generated[1], hidden[1], rtol=0, atol=0)


def test_gate_position_has_explicit_checkpoint_compatibility(tmp_path):
    config = config_planned(reviewed=True)
    path = tmp_path / "reviewed.pt"
    model = EviSeqAFMR(config)
    save_checkpoint(path, model, None, config, epoch=1, step=2)
    load_checkpoint(path, EviSeqAFMR(config), config=config)
    old_gate = copy.deepcopy(config)
    old_gate["decoder"]["grounded_copy"]["semantic_read"]["head_gate_position"] = "pre_norm"
    with pytest.raises(ValueError, match="architecture_spec"):
        load_checkpoint(path, EviSeqAFMR(old_gate), config=old_gate)
    old_partition = copy.deepcopy(config)
    old_partition["decoder"]["grounded_copy"]["semantic_read"]["planner"]["partition_heads"] = True
    with pytest.raises(ValueError, match="architecture_spec"):
        load_checkpoint(path, EviSeqAFMR(old_partition), config=old_partition)
    absent = copy.deepcopy(old_gate)
    del absent["decoder"]["grounded_copy"]["semantic_read"]["head_gate_position"]
    assert architecture_spec(absent) == architecture_spec(old_gate)


def test_recipe_selects_reviewed_graph_and_rejects_invalid_gate_position():
    from pathlib import Path

    config = load_config(Path(__file__).parents[1] / "configs/afmr_pubmed.yaml")
    semantic = config["decoder"]["grounded_copy"]["semantic_read"]
    assert semantic["rank"] // semantic["num_heads"] == 128
    assert semantic["head_gate_position"] == "post_norm"
    assert semantic["planner"]["partition_heads"] is False
    semantic["head_gate_position"] = "unknown"
    with pytest.raises(ValueError, match="head_gate_position"):
        validate_config(config)
