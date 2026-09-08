"""V3 source heads: independent oracle, CE gradients, cache and graph identity."""

import copy
import math
from dataclasses import fields

import pytest
import torch
import torch.nn.functional as F
from eviseq_update_v3.config import validate_config
from eviseq_update_v3.modeling.grounded_copy import CopyState, GroundedCopyHead
from eviseq_update_v3.modeling.model import EviSeqAFMR
from eviseq_update_v3.runtime import build_loaders
from eviseq_update_v3.training.checkpoint import architecture_spec, load_checkpoint, save_checkpoint
from eviseq_update_v3.training.optimizer import build_optimizer, set_stage_trainability
from test_semantic_read import model_config
from test_semantic_read_v2 import problem as v2_problem


def problem(active=True):
    _, lm, embedding, hidden, memory, bias, mask, alignment, _ = v2_problem()
    head = GroundedCopyHead(
        8,
        4,
        0.15,
        semantic_read=True,
        semantic_rank=8,
        semantic_num_heads=4,
        semantic_attention="independent_source",
        semantic_max_relative_rms=0.1,
    )
    if active:
        with torch.no_grad():
            head.semantic_output.weight.normal_(std=0.2)
    state = head.prepare(memory, bias, mask, embedding, **alignment)
    return head, lm, embedding, hidden, memory, bias, state


def config_v3():
    config = model_config("independent_bounded")
    config["decoder"]["query_cross_gate"] = True
    config["decoder"]["grounded_copy"]["semantic_read"].update(rank=8, num_heads=4)
    return config


def test_sdpa_context_and_gradients_match_independent_per_head_oracle():
    head, _, _, hidden, memory, bias, state = problem()
    query = head.semantic_query(head._norm(hidden)).float()
    contexts = []
    for i in range(4):
        sl = slice(i * 2, (i + 1) * 2)
        scores = query[..., sl] @ state.semantic_keys[..., sl].transpose(1, 2) / math.sqrt(2)
        valid = state.semantic_mask[:, None, :]
        probability = (scores + state.semantic_bias[:, None, :]).masked_fill(~valid, -1e30).softmax(-1)
        contexts.append(probability.masked_fill(~valid, 0) @ state.semantic_values[..., sl].float())
    expected = torch.cat(contexts, -1)
    actual = head._native_context(query, state)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    parameters = [
        hidden,
        memory,
        bias,
        head.semantic_query.weight,
        head.semantic_key.weight,
        head.semantic_value.weight,
    ]
    direction = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad((expected * direction).sum(), parameters, retain_graph=True)
    actual_gradients = torch.autograd.grad((actual * direction).sum(), parameters)
    for got, want in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
    assert actual[1].eq(0).all()


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
def test_multihead_dense_chunked_gold_ce_and_all_gradients_match(autocast, chunk_size):
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        head, lm, embedding, hidden, memory, bias, state = problem()
        labels = torch.tensor([[-100, -100, 4, 12, 5], [-100, -100, -100, 7, 11]])
        logits = head.output_logits(hidden, state, lm)
        expected = F.cross_entropy(logits.reshape(-1, 13), labels.reshape(-1))
        actual = head.loss(hidden, labels, state, lm, chunk_size)
    parameters = [hidden, memory, bias, embedding.weight, *head.parameters(), *lm.parameters()]
    expected_gradients = torch.autograd.grad(expected, parameters, retain_graph=True)
    actual_gradients = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, expected, atol=3e-3 if autocast else 2e-6, rtol=1e-4)
    for got, want in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, atol=2e-3 if autocast else 2e-6, rtol=2e-2 if autocast else 1e-4)
    assert actual_gradients[1][1].eq(0).all() and actual_gradients[1][0, 4].eq(0).all()


@pytest.mark.parametrize("autocast", [False, True])
def test_ce_updates_all_heads_and_gates_in_both_training_stages(autocast):
    config = config_v3()
    model = EviSeqAFMR(config).train()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    for stage in ("interface_warmup", "full_finetune"):
        set_stage_trainability(model, stage)
        optimizer = build_optimizer(model, config, stage)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                model(**tensors, return_logits=False).loss.backward()
            active = {
                name: p for name, p in model.named_parameters() if ".semantic_" in name or ".cross.query_gate." in name
            }
            for name, parameter in active.items():
                assert parameter.grad is not None and parameter.grad.dtype == torch.float32
                assert torch.isfinite(parameter.grad).all()
                if step:
                    assert parameter.grad.abs().sum() > 0, name
                    if name.endswith(("semantic_query.weight", "semantic_key.weight", "semantic_value.weight")):
                        assert parameter.grad.reshape(4, -1).abs().sum(-1).gt(0).all(), name
            optimizer.step()
            if step:
                for name, parameter in active.items():
                    assert not torch.equal(before[name], parameter), name


@pytest.mark.parametrize("width", [0, 3])
def test_multihead_empty_source_exact_fallback_and_zero_label_backward(width):
    head, lm, _, hidden, _, _, _ = problem()
    state = CopyState(
        torch.randn(2, width, 4),
        torch.zeros(2, width, dtype=torch.long),
        torch.zeros(2, width, dtype=torch.bool),
        torch.zeros(2, width),
        torch.randn(2, width, 8),
        torch.randn(2, width, 8),
        torch.zeros(2, width, dtype=torch.bool),
        torch.zeros(2, width),
    )
    torch.testing.assert_close(head.output_logits(hidden, state, lm), lm(hidden), rtol=0, atol=0)
    loss = head.loss(hidden, torch.full((2, 5), -100), state, lm, 3)
    assert loss.item() == 0
    loss.backward()
    # This manually supplied cache intentionally bypasses source projections.
    # The live decoder-side paths must still backpropagate finite zero gradients.
    for p in (
        hidden,
        head.query.weight,
        head.semantic_query.weight,
        head.semantic_output.weight,
        head.semantic_gate.weight,
    ):
        assert p.grad is not None and torch.isfinite(p.grad).all()


@pytest.mark.parametrize("autocast", [False, True])
def test_v3_learned_graph_cached_compaction_matches_full_prefix(autocast):
    config = config_v3()
    model = EviSeqAFMR(config).eval()
    with torch.no_grad():
        model.decoder.grounded_copy.semantic_output.weight.normal_(std=0.2)
        for layer in model.decoder.backbone.layers:
            layer.cross.query_gate.weight.normal_(std=0.15)
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        state = model.encode_source(
            output_budget=torch.full((tensors["input_ids"].shape[0],), float(config["generation"]["max_new_tokens"])),
            **{
                key: value
                for key, value in tensors.items()
                if key not in {"decoder_input_ids", "decoder_attention_mask", "labels"}
            },
        )
        decoder = model.decoder
        tokens = torch.tensor([[3, 4, 5], [6, 7, 8]])
        decoder.prepare_cross_cache(state.memory, state.value_memory)
        kwargs = dict(value_memory=state.value_memory, copy_state=state.copy_state)
        _, cache, _ = decoder(tokens, state.memory, state.memory_mask, state.source_bias, use_cache=True, **kwargs)
        rows = torch.tensor([1, 0, 1])
        selected_copy = state.copy_state.index_select(rows)
        for field in fields(state.copy_state):
            torch.testing.assert_close(getattr(selected_copy, field.name), getattr(state.copy_state, field.name)[rows])
        cache.batch_select_indices(rows)
        decoder.select_cross_cache(rows)
        next_ids = torch.tensor([[9], [10], [11]])
        kwargs = dict(value_memory=state.value_memory[rows], copy_state=selected_copy)
        got, _, _ = decoder(
            next_ids,
            state.memory[rows],
            state.memory_mask[rows],
            state.source_bias[rows],
            use_cache=True,
            past_key_values=cache,
            **kwargs,
        )
        decoder.clear_cross_cache()
        expected, _, _ = decoder(
            torch.cat((tokens[rows], next_ids), 1),
            state.memory[rows],
            state.memory_mask[rows],
            state.source_bias[rows],
            **kwargs,
        )
    torch.testing.assert_close(got[:, -1], expected[:, -1], atol=3e-3 if autocast else 1e-6, rtol=1e-3)


def test_v3_checkpoint_records_same_shape_architecture_and_rejects_other_head_count(tmp_path):
    config = config_v3()
    path = tmp_path / "v3.pt"
    model = EviSeqAFMR(config)
    save_checkpoint(path, model, None, config, epoch=1, step=2)
    restored = EviSeqAFMR(config)
    load_checkpoint(path, restored, config=config)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for heads, gate in ((1, True), (2, True), (4, False)):
        changed = copy.deepcopy(config)
        changed["decoder"]["query_cross_gate"] = gate
        changed["decoder"]["grounded_copy"]["semantic_read"]["num_heads"] = heads
        with pytest.raises(ValueError, match="architecture_spec"):
            load_checkpoint(path, EviSeqAFMR(changed), config=changed)
    legacy = model_config("independent_bounded")
    explicit = copy.deepcopy(legacy)
    explicit["decoder"]["query_cross_gate"] = False
    explicit["decoder"]["grounded_copy"]["semantic_read"]["num_heads"] = 1
    assert architecture_spec(legacy) == architecture_spec(explicit)


@pytest.mark.parametrize("heads", [0, -1, 3, 1.5, True, "4"])
def test_invalid_head_count_rejected(heads):
    config = config_v3()
    config["decoder"]["grounded_copy"]["semantic_read"]["num_heads"] = heads
    with pytest.raises(ValueError, match="num_heads"):
        validate_config(config)


def test_shared_copy_cannot_silently_ignore_multihead_setting():
    config = config_v3()
    config["decoder"]["grounded_copy"]["semantic_read"]["attention"] = "shared_copy"
    with pytest.raises(ValueError, match="independent_source"):
        validate_config(config)


def test_active_v3_cannot_read_future_target_tokens():
    config = config_v3()
    model = EviSeqAFMR(config).eval()
    with torch.no_grad():
        model.decoder.grounded_copy.semantic_output.weight.normal_(std=0.2)
        for layer in model.decoder.backbone.layers:
            layer.cross.query_gate.weight.normal_(std=0.15)
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    changed = {key: value.clone() for key, value in tensors.items()}
    # Keep source, prompt and decoder prefix fixed while replacing a suffix.
    start = tensors["decoder_input_ids"].shape[1] // 2
    changed["decoder_input_ids"][:, start:] = (changed["decoder_input_ids"][:, start:] + 7) % 128
    changed["labels"].fill_(-100)
    with torch.no_grad():
        before = model(**tensors).logits
        after = model(**changed).logits
    torch.testing.assert_close(after[:, :start], before[:, :start], rtol=0, atol=0)
    assert not torch.equal(after[:, start:], before[:, start:])


def test_multihead_layout_alone_does_not_guarantee_different_source_focus():
    head, _, _, hidden, _, _, state = problem()
    # Legitimate equal heads can attend identically. Diversity/coverage is an
    # empirical question; splitting rank128 into four heads does not enforce it.
    key = torch.randn(2, 5, 2)
    value = torch.randn(2, 5, 2)
    state.semantic_keys = key.repeat(1, 1, 4)
    state.semantic_values = value.repeat(1, 1, 4)
    query = torch.randn(2, hidden.shape[1], 2).repeat(1, 1, 4)
    contexts = head._native_context(query, state).reshape(2, hidden.shape[1], 4, 2)
    for index in range(1, 4):
        torch.testing.assert_close(contexts[:, :, index], contexts[:, :, 0], rtol=0, atol=0)
