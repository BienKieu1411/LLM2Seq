import copy
from pathlib import Path

import pytest
import torch

from eviseq_afmr.config import load_config, validate_config
from eviseq_afmr.modeling.decoder import ConditionalLowRankModulator
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import build_loaders
from eviseq_afmr.training.checkpoint import architecture_spec


ROOT = Path(__file__).parents[1]


def _config() -> dict:
    return load_config(ROOT / "configs" / "hyper_operator_smoke.yaml")


def _tensor_batch(config: dict) -> dict[str, torch.Tensor]:
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    return {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}


def test_zero_output_hyper_graph_exactly_matches_direct_projection_anchor():
    config = _config()
    batch = _tensor_batch(config)
    direct_config = copy.deepcopy(config)
    direct_config["architecture"]["bridge_mode"] = "direct_projection"
    validate_config(direct_config)

    torch.manual_seed(113)
    full = EviSeqAFMR(config).eval()
    torch.manual_seed(113)
    direct = EviSeqAFMR(direct_config).eval()
    with torch.no_grad():
        full_output = full(**batch)
        direct_output = direct(**batch)
    torch.testing.assert_close(full_output.logits, direct_output.logits, rtol=0, atol=0)
    torch.testing.assert_close(full_output.loss, direct_output.loss, rtol=0, atol=0)
    assert full_output.bridge.value_memory is not None
    torch.testing.assert_close(full_output.bridge.memory, full_output.bridge.value_memory, rtol=0, atol=0)


def test_low_rank_experts_open_gradient_routes_in_stages_and_change_output():
    torch.manual_seed(7)
    module = ConditionalLowRankModulator(
        input_dim=8,
        output_dim=6,
        controller_dim=5,
        num_experts=3,
        rank=4,
        gate_init=0.05,
        gate_max=0.20,
        max_relative_rms=0.10,
    )
    inputs = torch.randn(2, 7, 8)
    base = torch.randn(2, 7, 6)
    controller = torch.randn(2, 5, requires_grad=True)
    optimizer = torch.optim.SGD([*module.parameters(), controller], lr=0.25)

    initial = module(inputs, base, controller).detach()
    torch.testing.assert_close(initial, base, rtol=0, atol=0)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = module(inputs, base, controller)
        output.square().mean().backward()
        if step == 0:
            assert module.up.grad is not None and module.up.grad.abs().sum() > 0
        if step == 1:
            assert module.down.grad is not None and module.down.grad.abs().sum() > 0
            assert module.router.weight.grad is not None and module.router.weight.grad.abs().sum() > 0
        if step == 2:
            assert controller.grad is not None and controller.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.allclose(module(inputs, base, controller).detach(), base)


def test_ce_reaches_document_controller_after_expert_opens():
    torch.manual_seed(3)
    config = _config()
    model = EviSeqAFMR(config).train()
    batch = _tensor_batch(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    first_loss = model(**batch, return_logits=False).loss
    assert first_loss is not None
    first_loss.backward()
    up_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if "hyper_key.up" in name and parameter.grad is not None
    )
    assert up_grad > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_loss = model(**batch, return_logits=False).loss
    assert second_loss is not None
    second_loss.backward()
    controller_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if name.startswith("bridge.controller.") and parameter.grad is not None
    )
    assert controller_grad > 0


def test_operator_residual_is_capped_relative_to_each_base_token():
    module = ConditionalLowRankModulator(8, 6, 5, 3, 4, 0.05, 0.20, 0.10)
    with torch.no_grad():
        module.up.normal_(std=100.0)
        module.router.weight.normal_()
    inputs = torch.randn(2, 9, 8)
    base = torch.randn(2, 9, 6)
    output = module(inputs, base, torch.randn(2, 5))
    residual_rms = (output.float() - base.float()).square().mean(-1).sqrt()
    base_rms = base.float().square().mean(-1).sqrt()
    assert torch.all(residual_rms <= 0.10 * base_rms + 2.0e-6)


def test_expert_accumulation_matches_dense_mixture():
    module = ConditionalLowRankModulator(8, 6, 5, 3, 4, 0.05, 0.20, 0.10)
    with torch.no_grad():
        module.up.normal_(std=0.2)
        module.router.weight.normal_()
    inputs = torch.randn(2, 7, 8)
    base = torch.randn(2, 7, 6)
    controller = torch.randn(2, 5)
    weights = torch.softmax(module.router(controller), dim=-1)
    reduced = torch.einsum("bli,eri->bler", inputs, module.down)
    expert_delta = torch.einsum("bler,eor->bleo", reduced, module.up)
    delta = module.gate_max * torch.sigmoid(module.gate_raw) * torch.einsum("be,bleo->blo", weights, expert_delta)
    radius = module.max_relative_rms * base.square().mean(-1, keepdim=True).sqrt()
    expected = base + radius * delta / (radius.square() + delta.square().mean(-1, keepdim=True) + 1.0e-12).sqrt()
    torch.testing.assert_close(module(inputs, base, controller), expected, rtol=1.0e-5, atol=1.0e-6)


def test_controller_conditioned_cache_matches_uncached_cross_attention():
    config = _config()
    model = EviSeqAFMR(config).eval()
    batch = _tensor_batch(config)
    with torch.no_grad():
        bridge = model.encode_source(
            batch["input_ids"],
            batch["attention_mask"],
            batch["source_content_mask"],
            batch["decoder_prompt_ids"],
            batch["decoder_prompt_mask"],
            torch.full((batch["input_ids"].shape[0],), 32.0),
        )
        tokens = batch["decoder_input_ids"][:, :3]
        uncached, _, _ = model.decoder(
            tokens,
            bridge.memory,
            bridge.memory_mask,
            bridge.source_bias,
            controller=bridge.controller,
            value_memory=bridge.value_memory,
        )
        model.decoder.prepare_cross_cache(
            bridge.memory,
            bridge.value_memory,
            controller=bridge.controller,
        )
        cached, _, _ = model.decoder(
            tokens,
            bridge.memory,
            bridge.memory_mask,
            bridge.source_bias,
            controller=bridge.controller,
            value_memory=bridge.value_memory,
        )
        model.decoder.clear_cross_cache()
    torch.testing.assert_close(cached, uncached, rtol=1.0e-5, atol=1.0e-6)


def test_config_and_checkpoint_record_the_new_graph():
    config = _config()
    spec = architecture_spec(config)
    assert spec["architecture"] == "afmr_hyper_operator"
    assert spec["hyper_operator"]["mechanism"] == "document_conditioned_layerwise_lowrank_kv"
    assert spec["hyper_operator"]["num_experts"] == 2
    assert spec["hyper_operator"]["rank"] == 4

    invalid = copy.deepcopy(config)
    invalid["architecture"]["hyper_operator"]["gate_init"] = 0.3
    with pytest.raises(ValueError, match="gate"):
        validate_config(invalid)

    supervised = copy.deepcopy(config)
    supervised["training"]["salience_loss_weight"] = 0.002
    with pytest.raises(ValueError, match="CE-only"):
        validate_config(supervised)
