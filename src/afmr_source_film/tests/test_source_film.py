from __future__ import annotations

import torch

from source_film.config import load_config
from source_film.modeling.decoder import SourceConditionedAdaptiveNorm
from source_film.modeling.model import EviSeqAFMR


def test_config_selects_source_film_and_ce_only():
    config = load_config("src/afmr_source_film/configs/afmr_smoke.yaml")
    assert config["architecture"]["name"] == "source_conditioned_adaptive_norm"
    assert config["architecture"]["bridge_mode"] == "source_film"
    assert config["training"]["salience_loss_weight"] == 0


def test_adaptive_norm_is_exact_identity_at_initialization_and_has_finite_grads():
    module = SourceConditionedAdaptiveNorm(24, 16, rank=8)
    hidden = torch.randn(2, 5, 24, requires_grad=True)
    controller = torch.randn(2, 16, requires_grad=True)
    output = module(hidden, controller)
    torch.testing.assert_close(output, hidden, rtol=0, atol=0)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_model_source_film_has_zero_init_parity_and_controller_gradient():
    config = load_config("src/afmr_source_film/configs/afmr_smoke.yaml")
    config["model"]["gradient_checkpointing"] = False
    model = EviSeqAFMR(config).eval()
    batch, source_length, prompt_length, target_length = 2, 7, 3, 5
    source = torch.randint(0, 128, (batch, source_length))
    source_mask = torch.ones_like(source, dtype=torch.bool)
    prompt = torch.randint(0, 128, (batch, prompt_length))
    prompt_mask = torch.ones_like(prompt, dtype=torch.bool)
    target = torch.randint(0, 128, (batch, target_length))
    target_mask = torch.ones_like(target, dtype=torch.bool)
    with torch.no_grad():
        bridge = model.encode_source(source, source_mask, source_mask, prompt, prompt_mask, torch.full((batch,), 32.0))
        model.decoder.set_source_controller(None)
        baseline = model.decoder(target, bridge.memory, bridge.memory_mask, bridge.source_bias, target_mask)[0]
        model.decoder.set_source_controller(bridge.controller)
        conditioned = model.decoder(target, bridge.memory, bridge.memory_mask, bridge.source_bias, target_mask)[0]
    torch.testing.assert_close(conditioned, baseline, rtol=0, atol=0)

    # Simulate one optimizer update of an output projection: controller signal
    # must then reach the bridge, while the initial forward remains parity-safe.
    first_film = next(layer.source_film for layer in model.decoder.backbone.layers if layer.source_film is not None)
    torch.nn.init.normal_(first_film.scale_up.weight, std=0.01)
    output = model(
        source,
        source_mask,
        source_mask,
        prompt,
        prompt_mask,
        target,
        target_mask,
        return_logits=True,
    )
    output.logits.float().mean().backward()
    controller_grad = sum(parameter.grad.abs().sum() for parameter in model.bridge.controller.parameters())
    assert float(controller_grad) > 0
