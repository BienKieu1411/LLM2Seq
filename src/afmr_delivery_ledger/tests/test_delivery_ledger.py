from pathlib import Path

import torch
import torch.nn.functional as F
from eviseq_afmr.config import load_config
from eviseq_afmr.modeling.delivery_ledger import SourceDeliveryLedger, pool_source_regions
from eviseq_afmr.modeling.grounded_copy import CopyState, GroundedCopyHead
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import build_loaders
from eviseq_afmr.training.optimizer import build_optimizer, set_stage_trainability


def _ledger(hidden_size=12):
    return SourceDeliveryLedger(
        hidden_size,
        {
            "rank": 6,
            "write_strength_init": 0.10,
            "write_strength_max": 0.25,
            "coverage_strength_init": 0.20,
            "coverage_strength_max": 1.0,
            "read_enabled": True,
            "read_strength_init": 0.05,
            "read_strength_max": 0.20,
            "copy_strength_init": 0.10,
            "copy_strength_max": 0.50,
            "cross_strength_init": 0.10,
            "cross_strength_max": 0.50,
        },
    )


def test_region_pooling_ignores_prompt_and_padding():
    memory = torch.arange(8 * 2, dtype=torch.float32).view(1, 8, 2)
    content = torch.tensor([[False, True, True, True, False, True, True, False]])
    regions, mask, ids = pool_source_regions(memory, content, width=2)
    assert ids.tolist() == [[-1, 0, 0, 1, -1, 1, 2, -1]]
    assert mask.tolist() == [[True, True, True, False]]
    torch.testing.assert_close(regions[0, 0], memory[0, 1:3].mean(0))
    torch.testing.assert_close(regions[0, 1], memory[0, [3, 5]].mean(0))
    torch.testing.assert_close(regions[0, 2], memory[0, 6])


def test_inactive_prompt_does_not_write_or_modify_hidden():
    torch.manual_seed(2)
    ledger = _ledger()
    hidden = torch.randn(2, 5, 12)
    regions = torch.randn(2, 4, 12)
    region_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    active = torch.tensor([[False, False, True, True, True], [False, True, True, False, False]])
    routed, prior, cross_prior, coverage = ledger(hidden, regions, region_mask, active)
    torch.testing.assert_close(routed[~active], hidden[~active], rtol=0, atol=0)
    assert prior[~active].eq(0).all()
    assert cross_prior[~active].eq(0).all()
    assert coverage.ge(0).all() and coverage.le(1).all()
    assert coverage[0, :3].gt(0).all() and coverage[0, 3].eq(0)
    assert coverage[1, :2].gt(0).all() and coverage[1, 2:].eq(0).all()


def test_full_sequence_matches_cached_steps_and_state_compaction():
    torch.manual_seed(3)
    ledger = _ledger().eval()
    hidden = torch.randn(3, 4, 12)
    regions = torch.randn(3, 5, 12)
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False], [True, True, True, True, False]]
    )
    full_hidden, full_prior, full_cross_prior, full_coverage = ledger(hidden, regions, mask)
    coverage = None
    step_hidden, step_prior, step_cross_prior = [], [], []
    for index in range(hidden.shape[1]):
        routed, prior, cross_prior, coverage = ledger(hidden[:, index : index + 1], regions, mask, coverage=coverage)
        step_hidden.append(routed)
        step_prior.append(prior)
        step_cross_prior.append(cross_prior)
    torch.testing.assert_close(torch.cat(step_hidden, 1), full_hidden)
    torch.testing.assert_close(torch.cat(step_prior, 1), full_prior)
    torch.testing.assert_close(torch.cat(step_cross_prior, 1), full_cross_prior)
    torch.testing.assert_close(coverage, full_coverage)
    ledger.set_coverage(full_coverage)
    ledger.set_cross_coverage(full_coverage)
    ledger.select_state(torch.tensor([2, 0]))
    torch.testing.assert_close(ledger.get_coverage(), full_coverage[[2, 0]])
    torch.testing.assert_close(ledger.get_cross_coverage(), full_coverage[[2, 0]])
    ledger.clear_state()
    assert ledger.get_coverage() is None
    assert ledger.get_cross_coverage() is None


def test_only_history_changes_readout_and_copy_prior():
    torch.manual_seed(31)
    ledger = _ledger()
    ledger.write_top_k = 1
    hidden = torch.randn(1, 8, 12)
    hidden[:, 1:] = hidden[:, :1]
    regions = torch.randn(1, 5, 12)
    mask = torch.ones(1, 5, dtype=torch.bool)
    routed, prior, cross_prior, coverage = ledger(hidden, regions, mask)
    torch.testing.assert_close(routed[:, 0], hidden[:, 0], rtol=0, atol=0)
    assert prior[:, 0].eq(0).all()
    assert cross_prior[:, 0].eq(0).all()
    assert coverage.max() > coverage.min()
    assert not torch.allclose(routed[:, -1], hidden[:, -1])
    assert prior[:, -1].abs().max() > 0
    assert cross_prior[:, -1].abs().max() > 0
    with torch.no_grad():
        ledger.coverage_raw.fill_(-1000.0)
    neutral_hidden, neutral_prior, neutral_cross_prior, _ = ledger(hidden, regions, mask)
    torch.testing.assert_close(neutral_hidden, hidden, rtol=0, atol=0)
    assert neutral_prior.eq(0).all()
    assert neutral_cross_prior.eq(0).all()


def test_disabled_readout_preserves_decoder_hidden():
    ledger = SourceDeliveryLedger(
        12,
        {
            "rank": 6,
            "write_strength_init": 0.05,
            "write_strength_max": 0.25,
            "coverage_strength_init": 1.5,
            "coverage_strength_max": 3.0,
            "copy_strength_init": 0.50,
            "copy_strength_max": 1.0,
            "cross_strength_init": 0.50,
            "cross_strength_max": 1.0,
            "read_enabled": False,
        },
    )
    hidden = torch.randn(1, 8, 12)
    regions = torch.randn(1, 3, 12)
    routed, copy_prior, cross_prior, _ = ledger(hidden, regions, torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(routed, hidden, rtol=0, atol=0)
    assert copy_prior[:, -1].abs().max() > 0
    assert cross_prior[:, -1].abs().max() > 0


def test_history_gate_can_support_continuation_or_region_switching():
    torch.manual_seed(8)
    ledger = _ledger()
    ledger.write_top_k = 1
    hidden = torch.randn(1, 12, 12)
    hidden[:, 1:] = hidden[:, :1]
    regions = torch.randn(1, 4, 12)
    mask = torch.ones(1, 4, dtype=torch.bool)
    _, copy_switch, cross_switch, _ = ledger(hidden, regions, mask)
    with torch.no_grad():
        ledger.copy_gate.bias.neg_()
        ledger.cross_gate.bias.neg_()
    _, copy_continue, cross_continue, _ = ledger(hidden, regions, mask)
    assert copy_switch[:, -1].abs().max() > 0
    assert cross_switch[:, -1].abs().max() > 0
    torch.testing.assert_close(copy_continue, -copy_switch)
    torch.testing.assert_close(cross_continue, -cross_switch)


def test_copy_prior_changes_occurrence_choice_without_changing_copy_equation():
    torch.manual_seed(4)
    head = GroundedCopyHead(8, 4, 0.10)
    hidden = torch.randn(1, 1, 8)
    state = CopyState(
        keys=torch.zeros(1, 3, 4),
        token_ids=torch.tensor([[5, 6, 7]]),
        mask=torch.ones(1, 3, dtype=torch.bool),
        bias=torch.zeros(1, 3),
        region_ids=torch.tensor([[0, 1, 2]]),
    )
    neutral = head.distribution(hidden, state, torch.zeros(1, 1, 3))[0].exp()
    favored = head.distribution(hidden, state, torch.tensor([[[0.0, 2.0, 0.0]]]))[0].exp()
    torch.testing.assert_close(neutral, torch.full_like(neutral, 1 / 3))
    assert favored[0, 0, 1] > favored[0, 0, 0]
    assert favored[0, 0, 1] > favored[0, 0, 2]


def test_ledger_has_finite_gradients_through_time_and_bfloat16():
    torch.manual_seed(5)
    ledger = _ledger()
    hidden = torch.randn(2, 6, 12, requires_grad=True)
    regions = torch.randn(2, 4, 12, requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        routed, prior, cross_prior, coverage = ledger(hidden, regions, mask)
        loss = routed[..., 0].mean() + prior.square().mean() + cross_prior.square().mean() + coverage.square().mean()
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert regions.grad is not None and torch.isfinite(regions.grad).all()
    for parameter in ledger.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_tiny_model_dense_and_chunked_copy_losses_match():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["decoder"]["grounded_copy"]["enabled"] = True
    config["decoder"]["delivery_ledger"]["enabled"] = True
    config["decoder"]["delivery_ledger"]["rank"] = 8
    config["decoder"]["delivery_ledger"]["region_width"] = 2
    config["decoder"]["ce_chunk_size"] = 3
    model = EviSeqAFMR(config).eval()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    parameters = tuple(model.decoder.delivery_ledger.parameters())
    dense = model(**tensors, return_logits=True).loss
    dense_gradients = torch.autograd.grad(dense, parameters)
    chunked = model(**tensors, return_logits=False).loss
    chunked_gradients = torch.autograd.grad(chunked, parameters)
    torch.testing.assert_close(dense, chunked, rtol=1e-5, atol=1e-6)
    for dense_gradient, chunked_gradient in zip(dense_gradients, chunked_gradients):
        torch.testing.assert_close(dense_gradient, chunked_gradient, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(F.relu(chunked))


def test_final_cross_attention_receives_causal_coverage_bias():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["decoder"]["grounded_copy"]["enabled"] = True
    config["decoder"]["delivery_ledger"]["enabled"] = True
    config["decoder"]["delivery_ledger"]["region_width"] = 2
    config["decoder"]["delivery_ledger"]["write_top_k"] = 1
    model = EviSeqAFMR(config).eval()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    cross_biases = []

    def capture_bias(_module, args):
        cross_biases.append(args[3].detach())

    handles = [layer.cross.register_forward_pre_hook(capture_bias) for layer in model.decoder.backbone.layers]
    with torch.no_grad():
        model(**tensors, return_logits=False)
    for handle in handles:
        handle.remove()
    assert cross_biases[0].ndim == 2
    assert cross_biases[-1].shape == (2, 1, tensors["decoder_input_ids"].shape[1], tensors["input_ids"].shape[1])
    source_bias = cross_biases[0][:, None, None, :]
    torch.testing.assert_close(cross_biases[-1][:, :, :1], source_bias)
    valid = torch.isfinite(source_bias.expand_as(cross_biases[-1]))
    difference = (cross_biases[-1] - source_bias).masked_fill(~valid, 0.0)
    assert difference[:, :, 1:].abs().max() > 0


def test_cross_and_copy_regions_follow_their_actual_memory_views():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["decoder"]["grounded_copy"]["enabled"] = True
    config["decoder"]["delivery_ledger"]["enabled"] = True
    config["decoder"]["delivery_ledger"]["region_width"] = 2
    model = EviSeqAFMR(config).eval()
    with torch.no_grad():
        model.bridge.feature_up.weight.fill_(0.1)
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad():
        bridge = model(**tensors, return_logits=False).bridge
    expected_cross, mask, ids = pool_source_regions(bridge.memory, bridge.content_mask, 2)
    expected_copy, _, _ = pool_source_regions(bridge.value_memory, bridge.content_mask, 2)
    torch.testing.assert_close(bridge.cross_region_states, expected_cross)
    torch.testing.assert_close(bridge.region_states, expected_copy)
    torch.testing.assert_close(bridge.region_mask, mask)
    torch.testing.assert_close(bridge.source_region_ids, ids)
    assert not torch.allclose(bridge.cross_region_states, bridge.region_states)


def test_full_teacher_forcing_matches_cached_cross_and_copy_routes():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["decoder"]["grounded_copy"]["enabled"] = True
    config["decoder"]["delivery_ledger"]["enabled"] = True
    config["decoder"]["delivery_ledger"]["region_width"] = 2
    model = EviSeqAFMR(config).eval()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad():
        full = model(**tensors, return_logits=True)
        bridge = full.bridge
        model.decoder.prepare_cross_cache(bridge.memory, bridge.value_memory)
        past = None
        try:
            for index in range(tensors["decoder_input_ids"].shape[1] - 1):
                step_logits, past, _ = model.decoder(
                    tensors["decoder_input_ids"][:, index : index + 1],
                    bridge.memory,
                    bridge.memory_mask,
                    bridge.source_bias,
                    tensors["decoder_attention_mask"][:, : index + 1],
                    past_key_values=past,
                    use_cache=True,
                    value_memory=bridge.value_memory,
                    copy_state=bridge.copy_state,
                    region_states=bridge.region_states,
                    cross_region_states=bridge.cross_region_states,
                    region_mask=bridge.region_mask,
                    source_region_ids=bridge.source_region_ids,
                )
                active = tensors["labels"][:, index + 1].ne(-100)
                torch.testing.assert_close(step_logits[active, -1], full.logits[active, index], atol=1e-4, rtol=1e-4)
        finally:
            model.decoder.clear_cross_cache()


def test_warmup_updates_every_ledger_parameter():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["decoder"]["grounded_copy"]["enabled"] = True
    config["decoder"]["delivery_ledger"]["enabled"] = True
    config["decoder"]["delivery_ledger"]["rank"] = 8
    config["decoder"]["delivery_ledger"]["region_width"] = 2
    model = EviSeqAFMR(config)
    set_stage_trainability(model, "interface_warmup")
    optimizer = build_optimizer(model, config, "interface_warmup")
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    before = {name: value.detach().clone() for name, value in model.decoder.delivery_ledger.named_parameters()}
    model(**tensors, return_logits=False).loss.backward()
    for parameter in model.decoder.delivery_ledger.parameters():
        assert parameter.requires_grad
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    optimizer.step()
    for name, parameter in model.decoder.delivery_ledger.named_parameters():
        assert not torch.equal(before[name], parameter)
