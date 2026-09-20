from __future__ import annotations

import copy
from pathlib import Path

import torch

from eviseq_afmr.config import load_config
from eviseq_afmr.modeling.afmr import LayerwiseCoupledDepthBridge
from eviseq_afmr.modeling.grounded_copy import GroundedCopyHead
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.modeling.outputs import EncoderState


ROOT = Path(__file__).parents[1]


def _bridge_config(mode: str = "depth_kv") -> dict:
    return {
        "name": "layerwise_coupled_depth_kv",
        "bridge_mode": mode,
        "controller_dim": 6,
        "depth_taps": 3,
        "depth_rank": 5,
        "depth_gate_init": 0.10,
        "depth_gate_max": 1.0,
        "residual_max_relative_rms": 0.25,
        "final_tap_bias": 1.5,
    }


def _bridge_inputs() -> tuple[EncoderState, torch.Tensor, torch.Tensor, torch.Tensor]:
    final = torch.randn(2, 7, 8)
    taps = (torch.randn_like(final), torch.randn_like(final), final)
    attention = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
    content = attention.clone()
    content[:, :1] = False
    state = EncoderState(final, taps, attention, content)
    prompt = torch.randn(2, 4, 10)
    prompt_mask = torch.tensor([[True] * 4, [True] * 3 + [False]])
    budget = torch.tensor([64.0, 96.0])
    return state, prompt, prompt_mask, budget


def test_initial_full_bridge_exactly_matches_direct_projection() -> None:
    state, prompt, prompt_mask, budget = _bridge_inputs()
    torch.manual_seed(17)
    full = LayerwiseCoupledDepthBridge(8, 10, 4, _bridge_config())
    torch.manual_seed(17)
    direct = LayerwiseCoupledDepthBridge(8, 10, 4, _bridge_config("direct_projection"))

    full_state = full(state, prompt, prompt_mask, budget)
    direct_state = direct(state, prompt, prompt_mask, budget)
    torch.testing.assert_close(full_state.memory, direct_state.memory, rtol=0, atol=0)
    assert full_state.layer_memories is not None and len(full_state.layer_memories) == 4
    for layer_memory in full_state.layer_memories:
        torch.testing.assert_close(layer_memory, direct_state.memory, rtol=0, atol=0)
    assert torch.count_nonzero(full_state.source_bias) == 0
    assert direct_state.layer_memories is None


def test_ce_gradient_updates_outputs_then_reaches_router() -> None:
    config = load_config(ROOT / "configs/afmr_smoke.yaml")
    torch.manual_seed(19)
    model = EviSeqAFMR(config)
    batch = {
        "input_ids": torch.randint(3, 100, (2, 10)),
        "attention_mask": torch.ones(2, 10, dtype=torch.bool),
        "source_content_mask": torch.ones(2, 10, dtype=torch.bool),
        "decoder_prompt_ids": torch.randint(3, 100, (2, 3)),
        "decoder_prompt_mask": torch.ones(2, 3, dtype=torch.bool),
        "decoder_input_ids": torch.randint(3, 100, (2, 6)),
        "decoder_attention_mask": torch.ones(2, 6, dtype=torch.bool),
        "labels": torch.randint(3, 100, (2, 6)),
    }
    optimizer = torch.optim.SGD(model.bridge.parameters(), lr=0.5)
    before = [output.weight.detach().clone() for output in model.bridge.depth_outputs]
    loss = model(**batch, return_logits=False).loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    for output in model.bridge.depth_outputs:
        gradient = output.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    optimizer.step()
    assert all(not torch.equal(old, output.weight) for old, output in zip(before, model.bridge.depth_outputs))

    optimizer.zero_grad(set_to_none=True)
    second_loss = model(**batch, return_logits=False).loss
    assert second_loss is not None
    second_loss.backward()
    route_gradient = model.bridge.depth_route_condition.weight.grad
    assert route_gradient is not None and torch.isfinite(route_gradient).all() and route_gradient.abs().sum() > 0


def test_layers_diverge_after_the_zero_outputs_are_perturbed() -> None:
    bridge = LayerwiseCoupledDepthBridge(8, 10, 4, _bridge_config()).eval()
    with torch.no_grad():
        for index, output in enumerate(bridge.depth_outputs):
            output.weight.normal_(mean=0.0, std=0.02 * (index + 1))
            bridge.depth_route_bias[index].copy_(torch.tensor([index * 0.7, -index * 0.3, 1.5]))
    state, prompt, prompt_mask, budget = _bridge_inputs()
    result = bridge(state, prompt, prompt_mask, budget)
    assert result.layer_memories is not None
    assert any(
        not torch.allclose(result.layer_memories[0], result.layer_memories[index])
        for index in range(1, len(result.layer_memories))
    )
    for memory in result.layer_memories:
        assert memory.shape == result.memory.shape and torch.isfinite(memory).all()


def test_grounded_copy_reads_only_the_untouched_final_state_anchor() -> None:
    bridge = LayerwiseCoupledDepthBridge(8, 10, 3, _bridge_config()).eval()
    with torch.no_grad():
        for output in bridge.depth_outputs:
            output.weight.normal_(std=0.2)
    state, prompt, prompt_mask, budget = _bridge_inputs()
    bridged = bridge(state, prompt, prompt_mask, budget)
    assert bridged.layer_memories is not None
    assert any(not torch.allclose(memory, bridged.memory) for memory in bridged.layer_memories)

    head = GroundedCopyHead(hidden_size=10, key_dim=4, gate_init=0.05)
    embedding = torch.nn.Embedding(32, 10)
    alignment = {
        "copy_token_ids": torch.tensor([[3, 4, 5], [6, 7, 0]]),
        "copy_token_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "copy_encoder_indices": torch.tensor([[1, 2, 3], [1, 2, 2]]),
        "copy_token_indices": torch.tensor([[0, 1, 2], [0, 1, 2]]),
        "copy_alignment_weights": torch.ones(2, 3),
    }
    actual = head.prepare(bridged.memory, bridged.source_bias, bridged.content_mask, embedding, **alignment)
    expected = head.prepare(
        bridge.base_projection(state.final.float()),
        bridged.source_bias,
        bridged.content_mask,
        embedding,
        **alignment,
    )
    torch.testing.assert_close(actual.keys, expected.keys, rtol=0, atol=0)


def test_layerwise_memory_cached_decode_matches_full_decode() -> None:
    config = load_config(ROOT / "configs/afmr_smoke.yaml")
    config = copy.deepcopy(config)
    config["model"]["gradient_checkpointing"] = False
    decoder = EviSeqAFMR(config).decoder.eval()
    batch, source, hidden = 2, 7, int(decoder.config.hidden_size)
    anchor = torch.randn(batch, source, hidden)
    layer_memories = tuple(
        anchor + 0.02 * (index + 1) * torch.randn_like(anchor) for index in range(len(decoder.backbone.layers))
    )
    source_mask = torch.ones(batch, source, dtype=torch.bool)
    source_bias = torch.zeros(batch, source)
    tokens = torch.tensor([[3, 4, 5], [6, 7, 8]])

    with torch.no_grad():
        full, _, _ = decoder(
            tokens,
            anchor,
            source_mask,
            source_bias,
            attention_mask=torch.ones_like(tokens, dtype=torch.bool),
            layer_memories=layer_memories,
        )
        decoder.prepare_cross_cache(anchor, layer_memories=layer_memories)
        _, cache, _ = decoder(
            tokens[:, :2],
            anchor,
            source_mask,
            source_bias,
            attention_mask=torch.ones(batch, 2, dtype=torch.bool),
            use_cache=True,
            layer_memories=layer_memories,
        )
        cached, _, _ = decoder(
            tokens[:, 2:],
            anchor,
            source_mask,
            source_bias,
            attention_mask=torch.ones(batch, 3, dtype=torch.bool),
            past_key_values=cache,
            use_cache=True,
            layer_memories=layer_memories,
        )
        decoder.clear_cross_cache()
    assert full is not None and cached is not None
    torch.testing.assert_close(cached[:, -1], full[:, -1], atol=1.0e-6, rtol=1.0e-5)
