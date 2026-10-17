"""Architecture-specific counterexamples to decoder-only likelihood shortcuts."""

import copy

import pytest
import torch
from eviseq_afmr.data.copy_alignment import COPY_INPUT_KEYS
from eviseq_afmr.modeling.model import EviSeqAFMR
from seam_grpo.data import trajectory
from seam_grpo.policy import Policy
from test_policy import tiny


@pytest.mark.parametrize("copy_enabled", [False, True])
def test_full_reference_cannot_reuse_actor_encoder_memory(copy_enabled):
    cfg, tokenizer, batch = tiny(copy_enabled)
    actor = EviSeqAFMR(cfg).eval()
    reference = copy.deepcopy(actor).requires_grad_(False).eval()
    actions = batch["decoder_input_ids"][:, -3:].clone()
    packed = trajectory(batch, actions, torch.ones_like(actions, dtype=torch.bool), 4)
    with torch.no_grad():
        reference.encoder.model.embed_tokens.weight.add_(
            0.1 * torch.randn_like(reference.encoder.model.embed_tokens.weight)
        )
        actor_logps = Policy(actor)(packed, temperature=0.7)
        reference_logps = Policy(reference)(packed, temperature=0.7)
        actor_state = actor.encode_source(
            batch["input_ids"],
            batch["attention_mask"],
            batch["source_content_mask"],
            batch["decoder_prompt_ids"],
            batch["decoder_prompt_mask"],
            packed["output_budget"],
            **{key: batch[key] for key in COPY_INPUT_KEYS if key in batch},
        )
        # Deliberately WRONG: frozen decoder + actor-produced conditional states.
        logits, _, _ = reference.decoder(
            packed["decoder_input_ids"],
            actor_state.memory,
            actor_state.memory_mask,
            actor_state.source_bias,
            packed["decoder_attention_mask"],
            value_memory=actor_state.value_memory,
            copy_state=actor_state.copy_state,
        )
        width = batch["decoder_prompt_ids"].shape[1]
        wrong_logps = torch.log_softmax(logits[:, width - 1 : width + 2].float() / 0.7, dim=-1)
        wrong_logps = wrong_logps.gather(-1, actions[..., None]).squeeze(-1)
        torch.testing.assert_close(wrong_logps, actor_logps, rtol=2e-5, atol=2e-5)
        assert (reference_logps - wrong_logps).abs().max() > 1e-3
        # Updating the actor encoder must not silently move the frozen reference.
        actor.encoder.model.embed_tokens.weight.add_(0.2 * torch.randn_like(actor.encoder.model.embed_tokens.weight))
        again = Policy(reference)(packed, temperature=0.7)
        torch.testing.assert_close(again, reference_logps, rtol=0, atol=0)
    assert all(not p.requires_grad and p.grad is None for p in reference.parameters())


def test_rl_gradient_reaches_h0_memory_bias_and_copy_state():
    cfg, tokenizer, batch = tiny(True)
    model = EviSeqAFMR(cfg).eval()
    captured = {}

    def retain_h0(module, inputs, state):
        state.final.retain_grad()
        captured["H0"] = state.final

    def retain_bridge(module, inputs, state):
        for name, tensor in (("M", state.memory), ("V", state.value_memory), ("source_bias", state.source_bias)):
            assert tensor is not None and tensor.requires_grad, name
            tensor.retain_grad()
            captured[name] = tensor

    def retain_copy(module, inputs, state):
        for name, tensor in (("copy_keys", state.keys), ("copy_bias", state.bias)):
            tensor.retain_grad()
            captured[name] = tensor
        return state

    encoder_hook = model.encoder.register_forward_hook(retain_h0)
    bridge_hook = model.bridge.register_forward_hook(retain_bridge)
    original_prepare = model.decoder.grounded_copy.prepare

    def prepare(*args, **kwargs):
        return retain_copy(None, None, original_prepare(*args, **kwargs))

    model.decoder.grounded_copy.prepare = prepare
    try:
        actions = batch["decoder_input_ids"][:, -3:].clone()
        packed = trajectory(batch, actions, torch.ones_like(actions, dtype=torch.bool), 4)
        logps = Policy(model, chunk_size=1)(packed, temperature=0.8)
        (-logps[0].sum() + 0.3 * logps[1].sum()).backward()
        assert set(captured) == {"H0", "M", "V", "source_bias", "copy_keys", "copy_bias"}
        for name, tensor in captured.items():
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all(), name
            assert tensor.grad.abs().sum() > 0, name
        assert all(layer.cross._cache is None for layer in model.decoder.decoder_layers)
    finally:
        encoder_hook.remove()
        bridge_hook.remove()
        model.decoder.grounded_copy.prepare = original_prepare
