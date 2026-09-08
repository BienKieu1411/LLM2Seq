import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from eviseq_update_v3.config import load_config, validate_config
from eviseq_update_v3.modeling.grounded_copy import CopyState, GroundedCopyHead
from eviseq_update_v3.modeling.model import EviSeqAFMR
from eviseq_update_v3.runtime import build_loaders
from eviseq_update_v3.training.checkpoint import load_checkpoint, save_checkpoint
from eviseq_update_v3.training.optimizer import build_optimizer, set_stage_trainability

ROOT = Path(__file__).parents[1]


def model_config(variant="shared_v1"):
    config = load_config(ROOT / "configs/afmr_smoke.yaml")
    config["decoder"]["grounded_copy"]["enabled"] = True
    config["decoder"]["grounded_copy"]["semantic_read"] = {"enabled": True, "rank": 6, "gate_init": 0.05}
    if variant == "independent_bounded":
        config["decoder"]["grounded_copy"]["semantic_read"].update(
            attention="independent_source",
            max_relative_rms=0.1,
        )
    config["decoder"]["ce_chunk_size"] = 7
    return config


def problem():
    torch.manual_seed(72)
    head = GroundedCopyHead(8, 4, 0.15, semantic_read=True, semantic_rank=3)
    embedding, lm = torch.nn.Embedding(13, 8), torch.nn.Linear(8, 13)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    memory = torch.randn(2, 5, 8, requires_grad=True)
    bias = torch.randn(2, 5, requires_grad=True)
    alignment = dict(
        copy_token_ids=torch.tensor([[4, 4, 5, 6], [7, 8, 9, 10]]),
        copy_token_mask=torch.tensor([[True, True, True, False], [False, False, False, False]]),
        copy_encoder_indices=torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
        copy_token_indices=torch.tensor([[0, 0, 1, 2, 3], [0, 0, 1, 2, 3]]),
        copy_alignment_weights=torch.tensor([[0.4, 0.6, 1.0, 1.0, 1.0]]).expand(2, -1),
    )
    state = head.prepare(memory, bias, torch.ones(2, 5, dtype=torch.bool), embedding, **alignment)
    return head, lm, embedding, hidden, memory, bias, state


@pytest.mark.parametrize("variant", ["shared_v1", "independent_bounded"])
def test_zero_output_preserves_copy_only_model_weights_logits_and_loss(variant):
    config = model_config(variant)
    torch.manual_seed(41)
    updated = EviSeqAFMR(config).eval()
    control_config = copy.deepcopy(config)
    control_config["decoder"]["grounded_copy"]["semantic_read"]["enabled"] = False
    torch.manual_seed(41)
    control = EviSeqAFMR(control_config).eval()
    for name, tensor in control.state_dict().items():
        torch.testing.assert_close(updated.state_dict()[name], tensor, rtol=0, atol=0)
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad():
        before, after = control(**tensors), updated(**tensors)
        torch.testing.assert_close(after.logits, before.logits, rtol=0, atol=0)
        torch.testing.assert_close(after.loss, before.loss, rtol=0, atol=0)
        torch.testing.assert_close(updated(**tensors, return_logits=False).loss, after.loss)


@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
def test_active_semantic_dense_and_checkpointed_ce_have_identical_gradients(chunk_size):
    head, lm, embedding, hidden, memory, bias, state = problem()
    with torch.no_grad():
        head.semantic_output.weight.normal_(std=0.1)
    labels = torch.tensor([[-100, -100, 4, 12, 5], [-100, -100, -100, 7, 11]])
    dense = head.output_logits(hidden, state, lm)
    expected = F.cross_entropy(dense.reshape(-1, 13), labels.reshape(-1))
    parameters = [hidden, memory, bias, embedding.weight, *head.parameters(), *lm.parameters()]
    expected_gradients = torch.autograd.grad(expected, parameters, retain_graph=True)
    actual = head.loss(hidden, labels, state, lm, chunk_size)
    actual_gradients = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=1e-4)
    # The masked source row and masked token cannot condition the vocabulary branch.
    memory_grad = actual_gradients[1]
    assert memory_grad[1].eq(0).all() and memory_grad[0, 4].eq(0).all()
    torch.testing.assert_close(dense[1], lm(hidden)[1], rtol=0, atol=0)


def test_noncopyable_target_opens_alignment_gradient_after_first_update():
    head, lm, _, hidden, _, _, _ = problem()
    keys = torch.randn(2, 4, 4, requires_grad=True)
    values = torch.randn(2, 4, 3, requires_grad=True)
    state = CopyState(
        keys,
        torch.tensor([[4, 5, 6, 7], [4, 5, 6, 7]]),
        torch.ones(2, 4, dtype=torch.bool),
        torch.zeros(2, 4),
        values,
    )
    labels = torch.full((2, 5), 12)  # Absent from all eligible source IDs.
    optimizer = torch.optim.SGD(head.parameters(), lr=0.5)
    head.loss(hidden, labels, state, lm, 4).backward()
    assert head.semantic_output.weight.grad.abs().sum() > 0
    assert head.query.weight.grad.eq(0).all() and keys.grad.eq(0).all()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    keys.grad = values.grad = None
    # Isolate the new path from the old gate's indirect alignment path.
    with torch.no_grad():
        head.gate.weight.zero_()
    head.loss(hidden, labels, state, lm, 4).backward()
    for parameter in (head.query.weight, keys, values, head.semantic_gate.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("width", [0, 3])
@pytest.mark.parametrize("gate_bias", [-1000.0, 1000.0])
def test_empty_source_is_exact_lm_fallback_even_with_active_extreme_gates(width, gate_bias):
    head = GroundedCopyHead(8, 4, 0.05, semantic_read=True, semantic_rank=3)
    with torch.no_grad():
        head.semantic_output.weight.normal_()
        head.semantic_gate.bias.fill_(gate_bias)
        head.gate.bias.fill_(gate_bias)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    lm = torch.nn.Linear(8, 13)
    state = CopyState(
        torch.randn(2, width, 4),
        torch.zeros(2, width, dtype=torch.long),
        torch.zeros(2, width, dtype=torch.bool),
        torch.zeros(2, width),
        torch.randn(2, width, 3),
    )
    actual = head.output_logits(hidden, state, lm)
    torch.testing.assert_close(actual, lm(hidden), rtol=0, atol=0)
    labels = torch.full((2, 3), -100)
    loss = head.loss(hidden, labels, state, lm, 2)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(hidden.grad).all()


def test_active_head_cannot_silently_use_unconditioned_logits_or_missing_values():
    head, lm, _, hidden, _, _, state = problem()
    with pytest.raises(ValueError, match="output_logits"):
        head.mix_logits(hidden, lm(hidden), state)
    state.semantic_values = None
    with pytest.raises(ValueError, match="cached source values"):
        head.output_logits(hidden, state, lm)


def test_attention_is_computed_once_and_chunked_vocabulary_sees_only_supervised_positions():
    head, lm, _, hidden, _, _, state = problem()
    calls = []
    hook = head.query.register_forward_hook(lambda *args: calls.append(1))
    head.output_logits(hidden, state, lm)
    hook.remove()
    assert len(calls) == 1
    sizes = []
    hook = lm.register_forward_pre_hook(lambda module, args: sizes.append(args[0].shape[0]))
    labels = torch.tensor([[-100, -100, 4, 12, 5], [-100, -100, -100, 7, 11]])
    with torch.no_grad():
        head.loss(hidden, labels, state, lm, 4)
    hook.remove()
    assert sum(sizes) == int(labels.ne(-100).sum())
    assert max(sizes) <= 4


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("variant", ["shared_v1", "independent_bounded"])
def test_semantic_modules_learn_in_warmup_and_full_stages_with_fp32_updates(autocast, variant):
    torch.manual_seed(33)
    config = model_config(variant)
    model = EviSeqAFMR(config)
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    head = model.decoder.grounded_copy
    for stage in ("interface_warmup", "full_finetune"):
        set_stage_trainability(model, stage)
        optimizer = build_optimizer(model, config, stage)
        optimized_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        assert all(id(parameter) in optimized_ids for parameter in head.parameters())
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                loss = model(**tensors, return_logits=False).loss
            loss.backward()
            for name, parameter in head.named_parameters():
                assert parameter.dtype == parameter.grad.dtype == torch.float32
                assert torch.isfinite(parameter.grad).all()
                if step or stage == "full_finetune" or name == "semantic_output.weight":
                    assert parameter.grad.abs().sum() > 0, name
            optimizer.step()
            if step:
                for name, parameter in head.named_parameters():
                    assert not torch.equal(before[name], parameter), name
        if stage == "interface_warmup":
            assert all(parameter.grad is None for parameter in model.encoder.parameters())
        else:
            assert any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in model.encoder.parameters()
            )


def test_checkpoint_roundtrip_rejects_copy_only_and_different_semantic_rank(tmp_path):
    config = model_config()
    model = EviSeqAFMR(config)
    with torch.no_grad():
        model.decoder.grounded_copy.semantic_output.weight.normal_(std=0.1)
    path = tmp_path / "updated.pt"
    save_checkpoint(path, model, None, config, epoch=1, step=2)
    restored = EviSeqAFMR(config)
    load_checkpoint(path, restored, config=config)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for change in ({"enabled": False}, {"rank": 7}):
        incompatible = copy.deepcopy(config)
        incompatible["decoder"]["grounded_copy"]["semantic_read"].update(change)
        with pytest.raises(ValueError, match="architecture_spec"):
            load_checkpoint(path, EviSeqAFMR(incompatible), config=incompatible)
    old = copy.deepcopy(config)
    old["decoder"]["grounded_copy"].pop("semantic_read")
    legacy_path = tmp_path / "copy_only.pt"
    save_checkpoint(legacy_path, EviSeqAFMR(old), None, old, epoch=1, step=2)
    with pytest.raises(ValueError, match="architecture_spec"):
        load_checkpoint(legacy_path, model, config=config)


@pytest.mark.parametrize("setting", [False, {"enabled": "yes"}, {"rank": 0}, {"gate_init": 0}, {"unused": 1}])
def test_invalid_semantic_config_is_rejected(setting):
    config = model_config()
    config["decoder"]["grounded_copy"]["semantic_read"] = setting
    with pytest.raises(ValueError):
        validate_config(config)


def test_main_recipe_enables_semantic_read_and_requires_copy():
    config = load_config(ROOT / "configs/afmr_pubmed.yaml")
    assert config["decoder"]["grounded_copy"]["semantic_read"]["enabled"] is True
    config["decoder"]["grounded_copy"]["enabled"] = False
    with pytest.raises(ValueError, match="requires"):
        validate_config(config)
