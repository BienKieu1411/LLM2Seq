"""The regional read must be a trainable, isolated addition to AFMR."""

import copy
from pathlib import Path

import torch

from eviseq_afmr.config import load_config, validate_config
from eviseq_afmr.evaluation.generate import generate_greedy
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import build_loaders
from eviseq_afmr.training.checkpoint import architecture_spec
from eviseq_afmr.training.optimizer import build_optimizer, set_stage_trainability


def _config(enabled: bool = True) -> dict:
    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    config["architecture"]["region_query"]["enabled"] = enabled
    config["model"]["gradient_checkpointing"] = False
    config["decoder"]["grounded_copy"]["enabled"] = True
    config["training"]["salience_loss_weight"] = 0.0
    validate_config(config)
    return config


def _batch(config: dict) -> dict:
    return next(iter(build_loaders(config, max_train_examples=2)["train"]))


def test_region_route_starts_from_exact_static_afmr_with_copy():
    dynamic_config = _config()
    static_config = _config(False)
    batch = _batch(dynamic_config)
    tensor_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}

    torch.manual_seed(113)
    dynamic = EviSeqAFMR(dynamic_config).eval()
    dynamic_rng = torch.get_rng_state()
    torch.manual_seed(113)
    static = EviSeqAFMR(static_config).eval()
    torch.testing.assert_close(dynamic_rng, torch.get_rng_state(), rtol=0, atol=0)

    with torch.no_grad():
        dynamic_output = dynamic(**tensor_inputs)
        static_output = static(**tensor_inputs)
    torch.testing.assert_close(dynamic_output.logits, static_output.logits, rtol=0, atol=0)
    torch.testing.assert_close(dynamic_output.loss_ce, static_output.loss_ce, rtol=0, atol=0)
    assert dynamic_output.bridge.region_keys is not None
    assert static_output.bridge.region_keys is None
    first_region = dynamic.decoder.backbone.layers[0].cross.region_read
    second_region = dynamic.decoder.backbone.layers[1].cross.region_read
    assert not torch.equal(first_region.q_proj.weight, second_region.q_proj.weight)
    assert "region_query" in architecture_spec(dynamic_config)
    assert "region_query" not in architecture_spec(static_config)


def test_direct_projection_region_route_starts_from_direct_and_gets_ce_gradient():
    direct_region_config = _config()
    direct_region_config["architecture"]["bridge_mode"] = "direct_projection"
    validate_config(direct_region_config)
    direct_config = copy.deepcopy(direct_region_config)
    direct_config["architecture"]["region_query"]["enabled"] = False
    validate_config(direct_config)
    batch = _batch(direct_region_config)
    tensor_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}

    torch.manual_seed(137)
    direct_region = EviSeqAFMR(direct_region_config).eval()
    dynamic_rng = torch.get_rng_state()
    torch.manual_seed(137)
    direct = EviSeqAFMR(direct_config).eval()
    torch.testing.assert_close(dynamic_rng, torch.get_rng_state(), rtol=0, atol=0)
    with torch.no_grad():
        dynamic_output = direct_region(**tensor_inputs)
        direct_output = direct(**tensor_inputs)
    torch.testing.assert_close(dynamic_output.logits, direct_output.logits, rtol=0, atol=0)
    assert dynamic_output.bridge.region_keys is not None
    region = direct_region.decoder.backbone.layers[0].cross.region_read
    direct_region.train()
    loss = direct_region(**tensor_inputs, return_logits=False).loss_ce
    loss.backward()
    assert region.out_proj.weight.grad is not None
    assert region.out_proj.weight.grad.abs().sum() > 0


def test_static_bridge_and_direct_control_share_exact_initial_logits():
    static_config = _config(False)
    direct_config = copy.deepcopy(static_config)
    direct_config["architecture"]["bridge_mode"] = "direct_projection"
    validate_config(direct_config)
    batch = _batch(static_config)
    tensor_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}

    torch.manual_seed(131)
    static = EviSeqAFMR(static_config).eval()
    static_rng = torch.get_rng_state()
    torch.manual_seed(131)
    direct = EviSeqAFMR(direct_config).eval()
    torch.testing.assert_close(static_rng, torch.get_rng_state(), rtol=0, atol=0)
    with torch.no_grad():
        static_output = static(**tensor_inputs)
        direct_output = direct(**tensor_inputs)
    torch.testing.assert_close(static_output.logits, direct_output.logits, rtol=0, atol=0)


def test_ce_updates_region_output_then_upstream_projections():
    config = _config()
    batch = _batch(config)
    tensor_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    torch.manual_seed(127)
    model = EviSeqAFMR(config).train()
    region = model.decoder.backbone.layers[0].cross.region_read
    assert region is not None

    first_loss = model(**tensor_inputs, return_logits=False).loss_ce
    first_loss.backward()
    assert region.out_proj.weight.grad is not None
    assert region.out_proj.weight.grad.abs().sum() > 0
    before = region.out_proj.weight.detach().clone()
    optimizer = torch.optim.SGD(region.parameters(), lr=0.1)
    optimizer.step()
    assert not torch.equal(before, region.out_proj.weight)

    optimizer.zero_grad(set_to_none=True)
    second_loss = model(**tensor_inputs, return_logits=False).loss_ce
    second_loss.backward()
    assert region.q_proj.weight.grad is not None
    assert region.q_proj.weight.grad.abs().sum() > 0
    assert region.k_proj.weight.grad is not None
    assert region.k_proj.weight.grad.abs().sum() > 0
    assert region.v_proj.weight.grad is not None
    assert region.v_proj.weight.grad.abs().sum() > 0


def test_region_cache_greedy_compaction_matches_full_batch():
    config = _config()
    config["model"]["dtype"] = "bfloat16"
    loader = build_loaders(config, max_train_examples=2)["train"]
    batch = next(iter(loader))
    model = EviSeqAFMR(config).eval()
    tokenizer = loader.collate_fn.decoder_tokenizer

    with torch.no_grad():
        _, compact = generate_greedy(model, batch, tokenizer, 4, compact_finished=True)
        _, full = generate_greedy(model, batch, tokenizer, 4, compact_finished=False)
    torch.testing.assert_close(compact, full, rtol=0, atol=0)
    assert all(layer.cross.region_read._cache is None for layer in model.decoder.backbone.layers)


def test_bf16_autocast_keeps_fp32_region_gradients():
    config = _config()
    batch = _batch(config)
    tensor_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    model = EviSeqAFMR(config).train()
    region = model.decoder.backbone.layers[0].cross.region_read

    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(**tensor_inputs, return_logits=False).loss_ce
    loss.backward()
    assert torch.isfinite(loss)
    assert region.out_proj.weight.dtype == torch.float32
    assert region.out_proj.weight.grad.dtype == torch.float32
    assert region.out_proj.weight.grad.abs().sum() > 0


def test_region_parameters_are_in_warmup_cross_optimizer():
    config = _config()
    model = EviSeqAFMR(config)
    set_stage_trainability(model, "interface_warmup")
    region = model.decoder.backbone.layers[0].cross.region_read
    assert all(parameter.requires_grad for parameter in region.parameters())
    optimizer = build_optimizer(model, config, "interface_warmup")
    cross_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        if group["name"] == "cross_attention"
        for parameter in group["params"]
    }
    assert {id(parameter) for parameter in region.parameters()} <= cross_parameters
