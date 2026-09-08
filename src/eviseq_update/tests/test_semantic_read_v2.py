"""Independent source read: branch isolation, bounded residual and real CE gradients."""

import copy
from dataclasses import fields

import pytest
import torch
import torch.nn.functional as F
from eviseq_update.config import validate_config
from eviseq_update.modeling.grounded_copy import CopyState, GroundedCopyHead
from eviseq_update.modeling.model import EviSeqAFMR
from eviseq_update.training.checkpoint import architecture_spec, load_checkpoint, save_checkpoint
from test_semantic_read import model_config


def problem():
    torch.manual_seed(72)
    head = GroundedCopyHead(
        8,
        4,
        0.15,
        semantic_read=True,
        semantic_rank=3,
        semantic_attention="independent_source",
        semantic_max_relative_rms=0.1,
    )
    with torch.no_grad():
        head.semantic_output.weight.normal_(std=0.2)
    embedding, lm = torch.nn.Embedding(13, 8), torch.nn.Linear(8, 13)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    memory = torch.randn(2, 5, 8, requires_grad=True)
    bias = torch.randn(2, 5, requires_grad=True)
    mask = torch.tensor([[True, True, True, True, False], [False] * 5])
    alignment = dict(
        copy_token_ids=torch.tensor([[4, 4, 5, 6], [7, 8, 9, 10]]),
        copy_token_mask=torch.ones(2, 4, dtype=torch.bool),
        copy_encoder_indices=torch.arange(5).expand(2, -1),
        copy_token_indices=torch.tensor([[0, 0, 1, 2, 3]]).expand(2, -1),
        copy_alignment_weights=torch.tensor([[0.4, 0.6, 1.0, 1.0, 1.0]]).expand(2, -1),
    )
    state = head.prepare(memory, bias, mask, embedding, **alignment)
    return head, lm, embedding, hidden, memory, bias, mask, alignment, state


@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
@pytest.mark.parametrize("autocast", [False, True])
def test_dense_and_chunked_ce_match_for_every_gradient(chunk_size, autocast):
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        head, lm, embedding, hidden, memory, bias, _, _, state = problem()
        labels = torch.tensor([[-100, -100, 4, 12, 5], [-100, -100, -100, 7, 11]])
        dense = head.output_logits(hidden, state, lm)
        expected = F.cross_entropy(dense.reshape(-1, 13), labels.reshape(-1))
        actual = head.loss(hidden, labels, state, lm, chunk_size)
        fallback = lm(hidden)
    parameters = [hidden, memory, bias, embedding.weight, *head.parameters(), *lm.parameters()]
    expected_gradients = torch.autograd.grad(expected, parameters, retain_graph=True)
    actual_gradients = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, expected, atol=3e-3 if autocast else 2e-6, rtol=1e-4)
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(
            actual_grad,
            expected_grad,
            atol=2e-3 if autocast else 2e-6,
            rtol=2e-2 if autocast else 1e-4,
        )
    assert actual_gradients[1][1].eq(0).all() and actual_gradients[1][0, 4].eq(0).all()
    torch.testing.assert_close(dense[1], fallback[1].float(), rtol=0, atol=0)


def test_semantic_lm_read_does_not_backpropagate_through_lexical_copy_attention():
    head, _, embedding, hidden, memory, bias, _, _, state = problem()
    # Isolate the vocabulary context path from the final mixture CE. Backbones
    # and the source prior still receive both branches' gradients in the model.
    conditioned, _ = head.read(hidden, state)
    (conditioned * torch.randn_like(conditioned)).sum().backward()
    for parameter in (head.query.weight, head.context_key.weight, head.lexical_key.weight, embedding.weight):
        assert parameter.grad is None
    for parameter in (
        memory,
        bias,
        head.semantic_key.weight,
        head.semantic_query.weight,
        head.semantic_value.weight,
        head.semantic_output.weight,
        head.semantic_gate.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_native_read_ignores_copy_tokenization_and_copy_distribution_ignores_semantic_weights():
    head, _, embedding, hidden, memory, bias, mask, alignment, state = problem()
    expected_hidden, expected_copy = head.read(hidden, state)
    # Another copy tokenization/alignment of the same encoder memory.
    changed = dict(alignment)
    changed.update(
        copy_token_ids=torch.tensor([[11, 12], [1, 2]]),
        copy_token_mask=torch.ones(2, 2, dtype=torch.bool),
        copy_token_indices=torch.tensor([[0, 0, 0, 1, 1]]).expand(2, -1),
    )
    another = head.prepare(memory, bias, mask, embedding, **changed)
    actual_hidden, _ = head.read(hidden, another)
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
    with torch.no_grad():
        for name, parameter in head.named_parameters():
            if name.startswith("semantic_"):
                parameter.normal_()
    updated = head.prepare(memory, bias, mask, embedding, **alignment)
    _, actual_copy = head.read(hidden, updated)
    for actual, expected in zip(actual_copy, expected_copy):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("zero_hidden", [False, True])
def test_residual_bound_survives_large_projection_and_zero_states(autocast, zero_hidden):
    head, _, _, hidden, _, _, _, _, state = problem()
    with torch.no_grad():
        head.semantic_output.weight.mul_(10000)
        head.semantic_gate.bias.fill_(1000)
    if zero_hidden:
        hidden = torch.zeros_like(hidden, requires_grad=True)
    if autocast:
        hidden = hidden.detach().to(torch.bfloat16).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual, _ = head.read(hidden, state)

    def rms(value):
        return value.float().square().mean(-1).sqrt()

    # BF16 rounds h + delta; test the observed bound with the rounding allowance.
    tolerance = 0.008 if autocast else 1e-6
    assert (rms(actual.float() - hidden.float()) <= (0.1 + tolerance) * rms(hidden) + 1e-7).all()
    torch.testing.assert_close(actual[1], hidden[1], rtol=0, atol=0)
    actual.float().square().sum().backward()
    assert torch.isfinite(hidden.grad).all()
    for parameter in head.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_no_copy_candidates_still_allows_native_semantic_read_and_cache_compaction():
    head, lm, _, hidden, _, _, _, _, state = problem()
    state.mask.zero_()
    conditioned, _ = head.read(hidden, state)
    assert not torch.equal(conditioned[0], hidden[0])
    torch.testing.assert_close(head.output_logits(hidden, state, lm), lm(conditioned), rtol=0, atol=0)
    indices = torch.tensor([1, 0, 1])
    selected = state.index_select(indices)
    for field in fields(state):
        torch.testing.assert_close(getattr(selected, field.name), getattr(state, field.name)[indices])


@pytest.mark.parametrize("width", [0, 3])
def test_empty_native_source_has_exact_fallback_and_finite_backward(width):
    head, lm, _, hidden, _, _, _, _, _ = problem()
    state = CopyState(
        torch.randn(2, width, 4),
        torch.zeros(2, width, dtype=torch.long),
        torch.zeros(2, width, dtype=torch.bool),
        torch.zeros(2, width),
        torch.randn(2, width, 3),
        torch.randn(2, width, 3),
        torch.zeros(2, width, dtype=torch.bool),
        torch.zeros(2, width),
    )
    torch.testing.assert_close(head.output_logits(hidden, state, lm), lm(hidden), rtol=0, atol=0)
    loss = head.loss(hidden, torch.full((2, 5), -100), state, lm, 3)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(hidden.grad).all()


def test_v2_checkpoint_roundtrip_and_rejection_of_changed_graph_or_bound(tmp_path):
    config = model_config("independent_bounded")
    model = EviSeqAFMR(config)
    path = tmp_path / "v2.pt"
    save_checkpoint(path, model, None, config, epoch=1, step=2)
    restored = EviSeqAFMR(config)
    metadata = load_checkpoint(path, restored, config=config)
    assert metadata["training_spec"]["effective_batch_size"] == (
        config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"]
    )
    assert metadata["training_spec"]["max_grad_norm"] == config["training"]["max_grad_norm"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for change in ({"attention": "shared_copy"}, {"max_relative_rms": None}, {"max_relative_rms": 0.2}):
        incompatible = copy.deepcopy(config)
        incompatible["decoder"]["grounded_copy"]["semantic_read"].update(change)
        with pytest.raises(ValueError, match="architecture_spec"):
            load_checkpoint(path, EviSeqAFMR(incompatible), config=incompatible)


def test_explicit_shared_uncapped_is_compatible_with_original_v1_metadata():
    old = model_config()
    explicit = copy.deepcopy(old)
    explicit["decoder"]["grounded_copy"]["semantic_read"].update(attention="shared_copy", max_relative_rms=None)
    assert architecture_spec(old) == architecture_spec(explicit)


@pytest.mark.parametrize(
    "change",
    [
        {"attention": "typo"},
        {"max_relative_rms": 0},
        {"max_relative_rms": -0.1},
        {"max_relative_rms": float("inf")},
        {"max_relative_rms": float("nan")},
    ],
)
def test_invalid_v2_configuration_is_rejected(change):
    config = model_config("independent_bounded")
    config["decoder"]["grounded_copy"]["semantic_read"].update(change)
    with pytest.raises(ValueError):
        validate_config(config)
