"""Hierarchical read, causal prefix coverage and scale-preserving LM fusion."""

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from eviseq_update_v3.config import validate_config
from eviseq_update_v3.modeling.grounded_copy import GroundedCopyHead
from eviseq_update_v3.modeling.model import EviSeqAFMR
from eviseq_update_v3.modeling.semantic_plan import ReadPlan
from eviseq_update_v3.runtime import build_loaders
from eviseq_update_v3.training.checkpoint import load_checkpoint, save_checkpoint
from eviseq_update_v3.training.optimizer import build_optimizer, set_stage_trainability
from test_semantic_read import model_config


def config_planned():
    config = model_config("independent_bounded")
    config["decoder"]["query_cross_gate"] = False
    config["decoder"]["grounded_copy"]["semantic_read"].update(
        rank=32,
        num_heads=4,
        attention="hierarchical_coverage",
        fusion="norm_preserving",
        planner={"region_size": 1},
    )
    return config


def problem(active=True):
    torch.manual_seed(132)
    head = GroundedCopyHead(
        8,
        4,
        0.05,
        semantic_read=True,
        semantic_rank=16,
        semantic_num_heads=4,
        semantic_attention="hierarchical_coverage",
        semantic_max_relative_rms=0.1,
        semantic_fusion="norm_preserving",
        semantic_planner={"region_size": 2},
    )
    if active:
        with torch.no_grad():
            head.semantic_output.weight.normal_(std=0.2)
    hidden = torch.randn(2, 6, 8, requires_grad=True)
    memory = torch.randn(2, 18, 8, requires_grad=True)
    bias = torch.randn(2, 18, requires_grad=True)
    embedding, lm = torch.nn.Embedding(32, 8), torch.nn.Linear(8, 32)
    mask = torch.ones(2, 18, dtype=torch.bool)
    mask[0, -1] = False
    mask[1] = False
    state = head.prepare(
        memory,
        bias,
        mask,
        embedding,
        copy_token_ids=torch.arange(3, 21).expand(2, -1),
        copy_token_mask=torch.ones(2, 18, dtype=torch.bool),
        copy_encoder_indices=torch.arange(18).expand(2, -1),
        copy_token_indices=torch.arange(18).expand(2, -1),
        copy_alignment_weights=torch.ones(2, 18),
    )
    state.prompt_lengths = torch.tensor([2, 2])
    summary = torch.tensor([[False, False, True, True, True, True], [False, False, True, True, False, False]])
    plan, _ = head.planner(hidden, state, summary)
    return head, lm, hidden, memory, bias, embedding, state, summary, plan


def test_partitioned_full_width_attention_matches_loop_oracle():
    head, _, hidden, memory, bias, _, state, _, plan = problem()
    q = head.semantic_query(head._norm(hidden)).float()
    actual = head._hierarchical_context(q, state, plan)
    contexts = []
    for h in range(4):
        sl = slice(4 * h, 4 * (h + 1))
        scores = q[..., sl] @ state.region_keys[..., sl].transpose(1, 2) / 2
        valid = state.region_mask[:, None, :] & (torch.arange(9) % 4).eq(h)[None, None, :]
        rp = head.planner.masked_softmax(scores + state.region_bias[:, None, :] + head.planner.bias(plan), valid)
        out = torch.zeros(2, 6, 4)
        for region in range(9):
            mask = state.semantic_mask & state.token_regions.eq(region)
            token_scores = q[..., sl] @ state.semantic_keys[..., sl].transpose(1, 2) / 2
            a = head.planner.masked_softmax(token_scores + state.semantic_bias[:, None, :], mask[:, None, :])
            out = out + rp[..., region, None] * (a @ state.semantic_values[..., sl].float())
        contexts.append(out)
    expected = torch.cat(contexts, -1)
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)
    parameters = [
        hidden,
        memory,
        bias,
        *head.planner.parameters(),
        head.semantic_query.weight,
        head.semantic_key.weight,
        head.semantic_value.weight,
    ]
    direction = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * direction).sum(), parameters, retain_graph=True)
    ge = torch.autograd.grad((expected * direction).sum(), parameters)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=3e-6, rtol=3e-5)


def test_region_ownership_prevents_identical_position_support():
    head, _, hidden, _, _, _, state, _, plan = problem()
    q = head.semantic_query(head._norm(hidden)).float()
    before = head._hierarchical_context(q, state, plan)
    changed = copy.copy(state)
    changed.semantic_values = state.semantic_values.clone()
    # Only region0, which belongs to head0. Other heads have no path to it.
    changed.semantic_values[:, :2] += 100
    after = head._hierarchical_context(q, changed, plan)
    assert not torch.equal(after[0, :, :4], before[0, :, :4])
    torch.testing.assert_close(after[..., 4:], before[..., 4:], rtol=0, atol=0)


def test_prefix_scan_matches_incremental_history_and_excludes_prompt_padding_future():
    head, _, hidden, _, _, _, state, summary, plan = problem()
    history = None
    for step in range(hidden.shape[1]):
        one, history = head.planner(hidden[:, step : step + 1], state, summary[:, step : step + 1], history)
        torch.testing.assert_close(one.coverage[:, 0], plan.coverage[:, step], atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(one.recent[:, 0], plan.recent[:, step], atol=2e-7, rtol=2e-6)
    assert plan.coverage[:, :2].eq(0).all() and plan.recent[:, :2].eq(0).all()
    changed = hidden.detach().clone()
    changed[:, 4:] += 100
    other, _ = head.planner(changed, state, summary)
    torch.testing.assert_close(other.coverage[:, :4], plan.coverage[:, :4], rtol=0, atol=0)
    rows = torch.tensor([1, 0, 1])
    torch.testing.assert_close(history.index_select(rows).coverage, history.coverage[rows])


def test_coverage_reduces_used_region_while_recent_state_favors_continuation():
    head, _, hidden, _, _, _, state, _, _ = problem()
    zero = torch.zeros(2, 6, 9)
    used = zero.clone()
    used[..., 0] = 80
    penalty = head.planner.bias(ReadPlan(used, zero))
    assert (penalty[..., 0] < penalty[..., 4]).all()
    continuity = zero.clone()
    continuity[..., 4] = 1
    bonus = head.planner.bias(ReadPlan(zero, continuity))
    assert (bonus[..., 4] > bonus[..., 0]).all()


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
def test_planned_dense_chunked_ce_and_all_gradients_match(autocast, chunk_size):
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        head, lm, hidden, memory, bias, embedding, state, _, plan = problem()
        labels = torch.tensor([[-100, 4, 5, 28, 12, 9], [-100, -100, 7, 25, -100, -100]])
        logits = head.output_logits(hidden, state, lm, plan)
        expected = F.cross_entropy(logits.reshape(-1, 32), labels.reshape(-1))
        actual = head.loss(hidden, labels, state, lm, chunk_size, plan)
    parameters = [hidden, memory, bias, embedding.weight, *head.parameters(), *lm.parameters()]
    ge = torch.autograd.grad(expected, parameters, retain_graph=True)
    ga = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, expected, atol=3e-3 if autocast else 2e-6, rtol=1e-4)
    for a, e in zip(ga, ge):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e, atol=3e-3 if autocast else 3e-6, rtol=3e-2 if autocast else 2e-4)
    assert ga[1][1].eq(0).all() and ga[1][0, -1].eq(0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("zero", [False, True])
def test_fusion_preserves_norm_and_bounds_direction_even_with_extreme_delta(dtype, zero):
    head, _, hidden, _, _, _, _, _, _ = problem()
    hidden = (torch.zeros_like(hidden) if zero else hidden.detach()).to(dtype).requires_grad_()
    delta = (10000 * torch.randn_like(hidden.float())).requires_grad_()
    fused = head._fuse(hidden, delta)

    def norm(x):
        return torch.linalg.vector_norm(x.float(), dim=-1)

    tolerance = 0.008 if dtype == torch.bfloat16 else 3e-6
    torch.testing.assert_close(norm(fused), norm(hidden), rtol=tolerance, atol=1e-7)
    assert (norm(fused.float() - hidden.float()) <= (0.1 + tolerance) * norm(hidden) + 1e-7).all()
    fused.float().square().sum().backward()
    assert torch.isfinite(hidden.grad).all() and torch.isfinite(delta.grad).all()


def test_zero_delta_identity_and_radial_delta_cannot_change_lm_scale():
    head, _, hidden, _, _, _, _, _, _ = problem()
    torch.testing.assert_close(head._fuse(hidden, torch.zeros_like(hidden)), hidden, rtol=0, atol=0)
    torch.testing.assert_close(head._fuse(hidden, 3 * hidden), hidden, rtol=2e-6, atol=2e-6)


def test_semantic_changes_leave_copy_distribution_exact_for_fixed_base_hidden():
    head, _, hidden, memory, bias, _, state, _, plan = problem()
    expected = head.distribution(hidden, state)
    with torch.no_grad():
        for name, p in head.named_parameters():
            if name.startswith(("semantic_", "planner.")):
                p.normal_()
    _, actual = head.read(hidden, state, plan)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("autocast", [False, True])
def test_model_cached_prefix_and_compaction_match_dense_with_live_planner(autocast):
    config = config_planned()
    model = EviSeqAFMR(config).eval()
    with torch.no_grad():
        model.decoder.grounded_copy.semantic_output.weight.normal_(std=0.1)
        model.decoder.grounded_copy.semantic_head_gate.weight.normal_(std=0.1)
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    keys = {
        k: v
        for k, v in batch.items()
        if isinstance(v, torch.Tensor) and k not in {"decoder_input_ids", "decoder_attention_mask", "labels"}
    }
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        state = model.encode_source(**keys, output_budget=torch.tensor([32.0, 32.0]))
        dec = model.decoder
        prefix = batch["decoder_input_ids"][:, :6]
        dec.prepare_cross_cache(state.memory, state.value_memory)
        _, cache, _ = dec(
            prefix,
            state.memory,
            state.memory_mask,
            state.source_bias,
            copy_state=state.copy_state,
            value_memory=state.value_memory,
            use_cache=True,
        )
        rows = torch.tensor([1, 0, 1])
        cache.batch_select_indices(rows)
        dec.select_cross_cache(rows)
        copy_state = state.copy_state.index_select(rows)
        kwargs = dict(copy_state=copy_state, value_memory=state.value_memory[rows])
        next_ids = torch.tensor([[6], [7], [8]])
        got, _, _ = dec(
            next_ids,
            state.memory[rows],
            state.memory_mask[rows],
            state.source_bias[rows],
            past_key_values=cache,
            use_cache=True,
            **kwargs,
        )
        dec.clear_cross_cache()
        assert dec._semantic_history is None
        expected, _, _ = dec(
            torch.cat((prefix[rows], next_ids), 1),
            state.memory[rows],
            state.memory_mask[rows],
            state.source_bias[rows],
            **kwargs,
        )
    torch.testing.assert_close(got[:, -1], expected[:, -1], atol=4e-3 if autocast else 2e-6, rtol=1e-3)


def test_planner_receives_ce_gradients_and_updates_after_zero_output_opens():
    config = config_planned()
    model = EviSeqAFMR(config).train()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    inputs = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
    for stage in ("interface_warmup", "full_finetune"):
        set_stage_trainability(model, stage)
        opt = build_optimizer(model, config, stage)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            loss = model(**inputs, return_logits=False).loss
            loss.backward()
            for name, p in model.decoder.grounded_copy.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                if step and (name.startswith("planner.") or name.startswith("semantic_")):
                    assert p.grad.abs().sum() > 0, name
            before = {name: p.detach().clone() for name, p in model.decoder.grounded_copy.named_parameters()}
            opt.step()
            if step:
                for name, p in model.decoder.grounded_copy.named_parameters():
                    if name.startswith("planner."):
                        assert not torch.equal(before[name], p), name


def test_checkpoint_rejects_changed_region_semantics_and_fusion(tmp_path):
    config = config_planned()
    model = EviSeqAFMR(config)
    path = tmp_path / "planned.pt"
    save_checkpoint(path, model, None, config, epoch=1, step=2)
    load_checkpoint(path, EviSeqAFMR(config), config=config)
    for change in ({"region_size": 2}, {"partition_heads": False}, {"coverage_scale": 16.0}):
        other = copy.deepcopy(config)
        other["decoder"]["grounded_copy"]["semantic_read"]["planner"].update(change)
        with pytest.raises(ValueError, match="architecture_spec"):
            load_checkpoint(path, EviSeqAFMR(other), config=other)


def test_observed_prefix_counts_are_correct_with_left_padding_and_unequal_prompts():
    config = config_planned()
    model = EviSeqAFMR(config).eval()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    keys = {
        k: v
        for k, v in batch.items()
        if isinstance(v, torch.Tensor) and k not in {"decoder_input_ids", "decoder_attention_mask", "labels"}
    }
    with torch.no_grad():
        state = model.encode_source(**keys, output_budget=torch.tensor([32.0, 32.0]))
        state.copy_state.prompt_lengths = torch.tensor([3, 2])
        dec = model.decoder
        dec.grounded_copy.semantic_output.weight.normal_(std=0.1)
        tokens = torch.tensor([[0, 0, 3, 4, 5, 7], [8, 9, 10, 11, 12, 13]])
        mask = torch.tensor([[False, False, True, True, True, True], [True] * 6])
        kwargs = dict(copy_state=state.copy_state, value_memory=state.value_memory)
        dec.prepare_cross_cache(state.memory, state.value_memory)
        _, past, _ = dec(tokens, state.memory, state.memory_mask, state.source_bias, mask, use_cache=True, **kwargs)
        # Initial usage gate=.5. Only one/four observed summary inputs count.
        torch.testing.assert_close(dec._semantic_history.coverage.sum(-1), torch.tensor([0.5, 2.0]))
        next_ids = torch.tensor([[14], [15]])
        full_mask = torch.cat((mask, torch.ones(2, 1, dtype=torch.bool)), 1)
        got, _, _ = dec(
            next_ids,
            state.memory,
            state.memory_mask,
            state.source_bias,
            full_mask,
            use_cache=True,
            past_key_values=past,
            **kwargs,
        )
        dec.clear_cross_cache()
        full, _, _ = dec(
            torch.cat((tokens, next_ids), 1), state.memory, state.memory_mask, state.source_bias, full_mask, **kwargs
        )
    torch.testing.assert_close(got[:, -1], full[:, -1], atol=2e-6, rtol=1e-5)


def test_planned_semantic_branch_does_not_modify_v2_trunk_or_copy_for_shared_weights():
    config = config_planned()
    control_config = copy.deepcopy(config)
    control_config["decoder"]["grounded_copy"]["semantic_read"].update(
        rank=8, num_heads=1, attention="independent_source", fusion="residual"
    )
    torch.manual_seed(42)
    model = EviSeqAFMR(config).eval()
    torch.manual_seed(42)
    control = EviSeqAFMR(control_config).eval()
    with torch.no_grad():
        for name, p in model.decoder.grounded_copy.named_parameters():
            if name.startswith(("semantic_", "planner.")):
                p.normal_(std=0.3)
    captured = []

    def capture(module, args, output):
        captured.append(output.last_hidden_state.detach())

    hooks = [item.decoder.backbone.register_forward_hook(capture) for item in (control, model)]
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    inputs = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        before, after = control(**inputs), model(**inputs)
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
    expected = control.decoder.grounded_copy.distribution(captured[0], before.bridge.copy_state)
    actual = model.decoder.grounded_copy.distribution(captured[1], after.bridge.copy_state)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_coverage_ablation_has_no_dependency_on_previous_usage_when_disabled():
    head, _, _, _, _, _, _, _, plan = problem()
    head.planner.use_coverage = head.planner.use_continuity = False
    assert head.planner.bias(plan).eq(0).all()


@pytest.mark.parametrize(
    "planner",
    [
        {"region_size": 0},
        {"partition_heads": "yes"},
        {"coverage_scale": 0},
        {"coverage_init": 3},
        {"continuity_max": float("nan")},
    ],
)
def test_invalid_planner_configuration_rejected(planner):
    config = config_planned()
    config["decoder"]["grounded_copy"]["semantic_read"]["planner"].update(planner)
    with pytest.raises(ValueError):
        validate_config(config)
