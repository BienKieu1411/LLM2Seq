import copy
from pathlib import Path

import pytest
import torch
from grounded_summary.config import load_config
from grounded_summary.evaluation.generate import generate_greedy
from grounded_summary.modeling.model import AFMRModel
from grounded_summary.runtime import build_loaders
from grounded_summary.training.checkpoint import load_checkpoint, save_checkpoint
from grounded_summary.training.engine import AFMRTrainer
from grounded_summary.training.optimizer import build_optimizer, set_stage_trainability


def _setup(region_router: bool = False):
    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    if region_router:
        config["architecture"]["region_router"].update(enabled=True, window_size=4, start_fraction=0.75)
    model = AFMRModel(config)
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    return config, model, batch


def _forward(model, batch):
    return model(**{key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}, return_logits=False)


def _bridge(model, batch):
    return model.encode_source(
        *(
            batch[key]
            for key in (
                "input_ids",
                "attention_mask",
                "source_content_mask",
                "decoder_prompt_ids",
                "decoder_prompt_mask",
            )
        ),
        torch.full((batch["input_ids"].shape[0],), 32.0),
    )


def test_anchor_preserves_value_content_while_keys_adapt():
    torch.manual_seed(17)
    _, model, batch = _setup()
    model.eval()
    with torch.no_grad():
        before = _bridge(model, batch)
        cross = model.decoder.backbone.layers[0].cross
        key_before, value_before = cross._memory_kv(before.memory, before.value_memory)
        model.bridge.feature_up.weight.normal_(std=0.5)
        model.bridge.depth_out.weight.normal_(std=0.5)
        after = _bridge(model, batch)
        key_after, value_after = cross._memory_kv(after.memory, after.value_memory)
    assert not torch.allclose(key_before, key_after)
    torch.testing.assert_close(value_before, value_after, rtol=0, atol=0)
    torch.testing.assert_close(before.value_memory, after.value_memory, rtol=0, atol=0)


def test_zero_initialized_anchor_matches_shared_memory():
    config, model, batch = _setup()
    legacy_config = copy.deepcopy(config)
    legacy_config["architecture"]["name"] = "afmr_shared_memory"
    legacy = AFMRModel(legacy_config)
    legacy.load_state_dict(model.state_dict())
    model.eval()
    legacy.eval()
    with torch.no_grad():
        torch.testing.assert_close(_forward(model, batch).loss, _forward(legacy, batch).loss, rtol=0, atol=0)


def test_ce_gradients_reach_anchor_encoder_and_retrieval_routes():
    torch.manual_seed(19)
    config, model, batch = _setup()
    set_stage_trainability(model, "full_finetune")
    optimizer = build_optimizer(model, config, "full_finetune")
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = _forward(model, batch)
        output.bridge.value_memory.retain_grad()
        output.loss.backward()
        anchor_grad = output.bridge.value_memory.grad
        assert anchor_grad is not None and torch.isfinite(anchor_grad).all() and anchor_grad.abs().sum() > 0
        for parameter in (
            model.bridge.feature_up.weight,
            model.bridge.depth_out.weight,
            model.bridge.focus_output.weight,
            model.decoder.backbone.layers[0].cross.k_proj.weight,
            model.decoder.backbone.layers[0].cross.v_proj.weight,
        ):
            assert (
                parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
            )
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
        if step:
            assert model.bridge.depth_content_score.weight.grad.abs().sum() > 0
            assert model.bridge.focus_query.weight.grad.abs().sum() > 0
        optimizer.step()


def test_anchored_cache_compaction_matches_uncached_decoder():
    _, model, _ = _setup()
    decoder = model.decoder.eval()
    keys, values = torch.randn(3, 7, 24), torch.randn(3, 7, 24)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[:, -2:] = False
    bias = torch.randn(3, 7)
    tokens = torch.tensor([[3, 4], [5, 6], [7, 8]])
    rows, next_ids = torch.tensor([0, 2]), torch.tensor([[9], [10]])
    with torch.no_grad():
        decoder.prepare_cross_cache(keys, values)
        _, cache, _ = decoder(tokens, keys, mask, bias, value_memory=values, use_cache=True)
        cache.batch_select_indices(rows)
        decoder.select_cross_cache(rows)
        cached, _, _ = decoder(
            next_ids,
            keys[rows],
            mask[rows],
            bias[rows],
            value_memory=values[rows],
            past_key_values=cache,
            use_cache=True,
        )
        decoder.clear_cross_cache()
        full, _, _ = decoder(
            torch.cat((tokens[rows], next_ids), dim=1), keys[rows], mask[rows], bias[rows], value_memory=values[rows]
        )
    torch.testing.assert_close(cached[:, -1], full[:, -1], atol=1e-6, rtol=1e-5)


def test_region_router_step_zero_matches_disabled_model():
    torch.manual_seed(23)
    _, disabled, batch = _setup()
    _, enabled, _ = _setup(region_router=True)
    enabled.load_state_dict(disabled.state_dict(), strict=False)
    disabled.eval()
    enabled.eval()
    with torch.no_grad():
        expected = _forward(disabled, batch)
        actual = _forward(enabled, batch)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.bridge.memory, expected.bridge.memory, rtol=0, atol=0)
    torch.testing.assert_close(actual.bridge.value_memory, expected.bridge.value_memory, rtol=0, atol=0)
    torch.testing.assert_close(actual.bridge.source_bias, expected.bridge.source_bias, rtol=0, atol=0)
    assert actual.bridge.region_memory is not None


def test_region_router_initialization_preserves_shared_seed():
    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    routed_config = copy.deepcopy(config)
    routed_config["architecture"]["region_router"].update(enabled=True, window_size=4, start_fraction=0.75)
    torch.manual_seed(31)
    disabled = AFMRModel(config)
    torch.manual_seed(31)
    routed = AFMRModel(routed_config)
    routed_state = routed.state_dict()
    for name, parameter in disabled.state_dict().items():
        if name in routed_state:
            torch.testing.assert_close(parameter, routed_state[name], rtol=0, atol=0)


def test_region_router_cached_generation_matches_full_generation():
    config, model, batch = _setup(region_router=True)
    tokenizer = build_loaders(config)["train"].collate_fn.decoder_tokenizer
    model.eval()
    with torch.no_grad():
        _, compact = generate_greedy(model, batch, tokenizer, 5, compact_finished=True)
        _, full = generate_greedy(model, batch, tokenizer, 5, compact_finished=False)
    torch.testing.assert_close(compact, full)


def test_region_router_model_gradient_reaches_route_and_region_memory():
    config, model, batch = _setup(region_router=True)
    model.train()
    optimizer = build_optimizer(model, config, "full_finetune")
    active = [layer.cross for layer in model.decoder.backbone.layers if layer.cross.region_router_enabled]
    assert active
    gate_before = [cross.region_router_gate.detach().clone() for cross in active]
    first = _forward(model, batch)
    assert first.bridge.region_memory is not None
    first.bridge.region_memory.retain_grad()
    first.loss.backward()
    for cross in active:
        assert cross.region_router_gate.grad is not None
        assert torch.isfinite(cross.region_router_gate.grad) and cross.region_router_gate.grad.abs() > 0
    optimizer.step()
    assert any(not torch.equal(cross.region_router_gate, before) for cross, before in zip(active, gate_before))
    model.zero_grad(set_to_none=True)
    second = _forward(model, batch)
    assert second.bridge.region_memory is not None
    second.bridge.region_memory.retain_grad()
    second.loss.backward()
    assert second.bridge.region_memory.grad is not None
    assert torch.isfinite(second.bridge.region_memory.grad).all()
    for cross in active:
        for parameter in (cross.region_router_query.weight, cross.region_router_key.weight):
            assert (
                parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
            )


def test_bf16_autocast_keeps_fp32_updates_and_checkpoint_gradients():
    _, model, batch = _setup()
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = _forward(model, batch)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    for parameter in (model.bridge.feature_up.weight, model.decoder.backbone.layers[0].cross.v_proj.weight):
        assert parameter.dtype == torch.float32
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_bf16_inference_keeps_value_cache_dtype_and_greedy_parity():
    config, _, batch = _setup()
    config["model"]["dtype"] = "bfloat16"
    model = AFMRModel(config).eval()
    tokenizer = build_loaders(config)["train"].collate_fn.decoder_tokenizer
    with torch.no_grad():
        bridge = _bridge(model, batch)
        assert bridge.memory.dtype == bridge.value_memory.dtype == torch.bfloat16
        model.decoder.prepare_cross_cache(bridge.memory, bridge.value_memory)
        for layer in model.decoder.backbone.layers:
            assert all(tensor.dtype == torch.bfloat16 for tensor in layer.cross._cache)
        model.decoder.clear_cross_cache()
        _, compact = generate_greedy(model, batch, tokenizer, 4, compact_finished=True)
        _, full = generate_greedy(model, batch, tokenizer, 4, compact_finished=False)
    torch.testing.assert_close(compact, full)


def test_bf16_region_router_forward_is_finite():
    config, _, batch = _setup(region_router=True)
    config["model"]["dtype"] = "bfloat16"
    model = AFMRModel(config).eval()
    with torch.no_grad():
        output = _forward(model, batch)
    assert output.bridge.region_memory is not None
    assert output.bridge.region_memory.dtype == torch.bfloat16
    assert torch.isfinite(output.loss)


def test_trainer_promotes_bf16_parameters_and_optimizer_states(tmp_path):
    config, model, _ = _setup()
    config["experiment"]["output_dir"] = str(tmp_path)
    trainer = AFMRTrainer(model.to(torch.bfloat16), config, "cpu")
    assert all(p.dtype == torch.float32 for p in trainer.model.parameters())
    set_stage_trainability(trainer.model, "full_finetune")
    optimizer = build_optimizer(trainer.model, config, "full_finetune")
    parameter = trainer.model.decoder.backbone.layers[0].cross.v_proj.weight
    with torch.no_grad():
        parameter.fill_(1.0)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.full_like(parameter, 0.01)
        optimizer.step()
    assert (parameter < 0.9999).all()
    state = optimizer.state[parameter]
    assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    save_checkpoint(tmp_path / "last.pt", trainer.model, optimizer, config, epoch=1, step=10)
    restored_model = AFMRModel(config)
    set_stage_trainability(restored_model, "full_finetune")
    restored_optimizer = build_optimizer(restored_model, config, "full_finetune")
    load_checkpoint(tmp_path / "last.pt", restored_model, restored_optimizer, config)
    assert all(state["exp_avg"].dtype == torch.float32 for state in restored_optimizer.state.values())
    legacy_config = copy.deepcopy(config)
    legacy_config["architecture"]["name"] = "afmr_shared_memory"
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(tmp_path / "last.pt", restored_model, config=legacy_config)
