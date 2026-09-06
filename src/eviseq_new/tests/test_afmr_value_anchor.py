import copy
from pathlib import Path

import pytest
import torch
from eviseq_afmr.config import load_config
from eviseq_afmr.evaluation.generate import generate_greedy
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import build_loaders
from eviseq_afmr.training.checkpoint import load_checkpoint, save_checkpoint
from eviseq_afmr.training.engine import AFMRTrainer
from eviseq_afmr.training.optimizer import build_optimizer, set_stage_trainability


def _setup():
    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    model = EviSeqAFMR(config)
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
    legacy_config["architecture"]["name"] = "afmr_v1"
    legacy = EviSeqAFMR(legacy_config)
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
    model = EviSeqAFMR(config).eval()
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
    restored_model = EviSeqAFMR(config)
    set_stage_trainability(restored_model, "full_finetune")
    restored_optimizer = build_optimizer(restored_model, config, "full_finetune")
    load_checkpoint(tmp_path / "last.pt", restored_model, restored_optimizer, config)
    assert all(state["exp_avg"].dtype == torch.float32 for state in restored_optimizer.state.values())
    legacy_config = copy.deepcopy(config)
    legacy_config["architecture"]["name"] = "afmr_v1"
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(tmp_path / "last.pt", restored_model, config=legacy_config)
