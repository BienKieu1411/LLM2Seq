"""Query-dependent head gates preserve initialization and learn through gold CE."""

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from eviseq_update_v3.config import load_config
from eviseq_update_v3.modeling.decoder import CopiedCrossAttention, QwenCrossDecoder
from eviseq_update_v3.modeling.model import EviSeqAFMR
from eviseq_update_v3.runtime import build_loaders
from eviseq_update_v3.training.optimizer import build_optimizer, set_stage_trainability


def attention_pair(dtype=torch.float32):
    torch.manual_seed(187)
    base = SimpleNamespace(
        q_proj=nn.Linear(8, 8, bias=False, dtype=dtype),
        k_proj=nn.Linear(8, 4, bias=False, dtype=dtype),
        v_proj=nn.Linear(8, 4, bias=False, dtype=dtype),
        o_proj=nn.Linear(8, 8, bias=False, dtype=dtype),
        q_norm=nn.Identity(),
        k_norm=nn.Identity(),
    )
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=4)
    control = CopiedCrossAttention(base, nn.Identity(), config, 0.0).eval()
    rng_before = torch.random.get_rng_state().clone()
    gated = CopiedCrossAttention(base, nn.Identity(), config, 0.0, query_cross_gate=True).eval()
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    return control, gated


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_gate_preserves_copied_weights_rng_and_attention_exactly(dtype):
    control, gated = attention_pair(dtype)
    for name, tensor in control.state_dict().items():
        torch.testing.assert_close(gated.state_dict()[name], tensor, rtol=0, atol=0)
    query = torch.randn(2, 4, 8, dtype=dtype)
    memory = torch.randn(2, 6, 8, dtype=dtype)
    values = torch.randn_like(memory)
    mask = torch.tensor([[True] * 4 + [False] * 2, [True] * 6])
    bias = torch.randn(2, 6)
    expected = control(query, memory, mask, bias, values)
    actual = gated(query, memory, mask, bias, values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_learned_gate_selects_different_heads_for_different_target_tokens():
    control, gated = attention_pair()
    with torch.no_grad():
        control.o_proj.weight.copy_(torch.eye(8))
        gated.o_proj.weight.copy_(torch.eye(8))
        gated.query_gate.weight[0, 0] = math.log(3.0)
        gated.query_gate.weight[1, 0] = -math.log(3.0)
    query = torch.zeros(1, 2, 8)
    query[0, :, 0] = torch.tensor([1.0, -1.0])
    memory = torch.randn(1, 5, 8)
    mask = torch.ones(1, 5, dtype=torch.bool)
    expected = control(query, memory, mask, None).reshape(1, 2, 2, 4)
    multiplier = torch.tensor([[[1.5, 0.5], [0.5, 1.5]]])[..., None]
    actual = gated(query, memory, mask, None).reshape(1, 2, 2, 4)
    torch.testing.assert_close(actual, expected * multiplier)
    assert not torch.allclose(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_active_gate_has_finite_live_gradients_with_masked_source(dtype):
    _, gated = attention_pair(dtype)
    with torch.no_grad():
        gated.query_gate.weight.normal_(std=0.1)
        gated.query_gate.bias.fill_(0.2)
    query = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    memory = torch.randn(2, 5, 8, dtype=dtype, requires_grad=True)
    bias = torch.randn(2, 5, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    output = gated(query, memory, mask, bias)
    (output.float() * torch.randn_like(output.float())).sum().backward()
    for value in (query, memory, bias, gated.query_gate.weight, gated.query_gate.bias):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert memory.grad[0, 3:].eq(0).all()


def decoder_config(enabled=True):
    return {"query_cross_gate": enabled, "grounded_copy": {"enabled": False}}


def test_enabling_query_gate_preserves_complete_decoder_initialization():
    torch.manual_seed(92)
    control = QwenCrossDecoder("__tiny__", decoder_config(False), torch.float32, False).eval()
    expected_random = torch.randn(5)
    torch.manual_seed(92)
    gated = QwenCrossDecoder("__tiny__", decoder_config(True), torch.float32, False).eval()
    torch.testing.assert_close(torch.randn(5), expected_random, rtol=0, atol=0)
    for name, value in control.state_dict().items():
        torch.testing.assert_close(gated.state_dict()[name], value, rtol=0, atol=0)
    tokens = torch.tensor([[3, 4, 5], [6, 7, 8]])
    memory = torch.randn(2, 5, 24)
    mask = torch.ones(2, 5, dtype=torch.bool)
    bias = torch.randn(2, 5)
    with torch.no_grad():
        before, _, before_loss = control(tokens, memory, mask, bias, labels=tokens)
        after, _, after_loss = gated(tokens, memory, mask, bias, labels=tokens)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    torch.testing.assert_close(after_loss, before_loss, rtol=0, atol=0)


@pytest.mark.parametrize("autocast", [False, True])
def test_learned_query_gate_prefix_and_compacted_cache_match_full_decode(autocast):
    torch.manual_seed(37)
    decoder = QwenCrossDecoder("__tiny__", decoder_config(), torch.float32, False).eval()
    with torch.no_grad():
        for layer in decoder.backbone.layers:
            layer.cross.query_gate.weight.normal_(std=0.15)
            layer.cross.query_gate.bias.normal_(std=0.1)
    tokens = torch.tensor([[3, 4, 5], [6, 7, 8], [9, 10, 11]])
    memory = torch.randn(3, 6, 24)
    values = torch.randn_like(memory)
    mask = torch.tensor([[True] * 5 + [False], [True] * 6, [True] * 4 + [False] * 2])
    bias = torch.randn(3, 6)
    rows = torch.tensor([2, 0])
    next_ids = torch.tensor([[12], [13]])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        decoder.prepare_cross_cache(memory, values)
        _, cache, _ = decoder(tokens, memory, mask, bias, value_memory=values, use_cache=True)
        cache.batch_select_indices(rows)
        decoder.select_cross_cache(rows)
        cached, _, _ = decoder(
            next_ids,
            memory[rows],
            mask[rows],
            bias[rows],
            value_memory=values[rows],
            past_key_values=cache,
            use_cache=True,
        )
        decoder.clear_cross_cache()
        full, _, _ = decoder(
            torch.cat((tokens[rows], next_ids), dim=1),
            memory[rows],
            mask[rows],
            bias[rows],
            value_memory=values[rows],
        )
    torch.testing.assert_close(cached[:, -1], full[:, -1], atol=2e-3 if autocast else 1e-6, rtol=1e-3)


@pytest.mark.parametrize("autocast", [False, True])
def test_gold_ce_updates_query_gates_in_both_stages_with_checkpointing(autocast):
    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    config["decoder"].update(query_cross_gate=True, grounded_copy={"enabled": False})
    config["model"]["gradient_checkpointing"] = True
    torch.manual_seed(38)
    model = EviSeqAFMR(config).train()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    for stage in ("interface_warmup", "full_finetune"):
        set_stage_trainability(model, stage)
        optimizer = build_optimizer(model, config, stage)
        query_parameters = {
            name: parameter for name, parameter in model.named_parameters() if ".cross.query_gate." in name
        }
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        assert len(query_parameters) == 2 * len(model.decoder.backbone.layers)
        assert all(id(parameter) in optimized and parameter.requires_grad for parameter in query_parameters.values())
        before = {name: parameter.detach().clone() for name, parameter in query_parameters.items()}
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            loss = model(**tensors, return_logits=False).loss
        loss.backward()
        for parameter in query_parameters.values():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        optimizer.step()
        for name, parameter in query_parameters.items():
            assert not torch.equal(parameter, before[name])
        model.zero_grad(set_to_none=True)


def test_checkpointed_decoder_matches_direct_gradients_with_active_query_gates():
    torch.manual_seed(54)
    direct = QwenCrossDecoder("__tiny__", decoder_config(), torch.float32, False).train()
    with torch.no_grad():
        for layer in direct.backbone.layers:
            layer.cross.query_gate.weight.normal_(std=0.1)
    checked = copy.deepcopy(direct)
    checked.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tokens = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    memory = torch.randn(2, 5, 24, requires_grad=True)
    other_memory = memory.detach().clone().requires_grad_()
    mask = torch.ones(2, 5, dtype=torch.bool)
    bias = torch.randn(2, 5)
    _, _, expected = direct(tokens, memory, mask, bias, labels=tokens, return_logits=False)
    _, _, actual = checked(tokens, other_memory, mask, bias, labels=tokens, return_logits=False)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(other_memory.grad, memory.grad, atol=1e-6, rtol=1e-5)
    for (name, parameter), (other_name, other) in zip(direct.named_parameters(), checked.named_parameters()):
        assert name == other_name
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(other.grad, parameter.grad, atol=1e-6, rtol=1e-5)
