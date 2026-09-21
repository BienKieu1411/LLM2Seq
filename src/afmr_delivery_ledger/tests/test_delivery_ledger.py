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
            "read_strength_init": 0.05,
            "read_strength_max": 0.20,
            "copy_strength_init": 0.10,
            "copy_strength_max": 0.50,
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
    routed, prior, coverage = ledger(hidden, regions, region_mask, active)
    torch.testing.assert_close(routed[~active], hidden[~active], rtol=0, atol=0)
    assert prior[~active].eq(0).all()
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
    full_hidden, full_prior, full_coverage = ledger(hidden, regions, mask)
    coverage = None
    step_hidden, step_prior = [], []
    for index in range(hidden.shape[1]):
        routed, prior, coverage = ledger(hidden[:, index : index + 1], regions, mask, coverage=coverage)
        step_hidden.append(routed)
        step_prior.append(prior)
    torch.testing.assert_close(torch.cat(step_hidden, 1), full_hidden)
    torch.testing.assert_close(torch.cat(step_prior, 1), full_prior)
    torch.testing.assert_close(coverage, full_coverage)
    ledger.set_coverage(full_coverage)
    ledger.select_state(torch.tensor([2, 0]))
    torch.testing.assert_close(ledger.get_coverage(), full_coverage[[2, 0]])
    ledger.clear_state()
    assert ledger.get_coverage() is None


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
    mask = torch.ones(2, 4, dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        routed, prior, coverage = ledger(hidden, regions, mask)
        loss = routed[..., 0].mean() + prior.square().mean() + coverage.square().mean()
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
    dense = model(**tensors, return_logits=True).loss
    chunked = model(**tensors, return_logits=False).loss
    torch.testing.assert_close(dense, chunked, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(F.relu(chunked))


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
