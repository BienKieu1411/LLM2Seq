from __future__ import annotations

import copy
from pathlib import Path

import torch

from afmr_side_memory.config import load_config
from afmr_side_memory.modeling.decoder import DecoderLayerWithCross
from afmr_side_memory.modeling.model import SideMemorySummarizer
from afmr_side_memory.training.optimizer import build_optimizer, set_stage_trainability


ROOT = Path(__file__).resolve().parents[1]


def config(*, copy_enabled: bool = False) -> dict:
    result = load_config(ROOT / "configs" / "afmr_smoke.yaml")
    result["model"]["gradient_checkpointing"] = False
    result["decoder"]["grounded_copy"]["enabled"] = copy_enabled
    return result


def inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    source = torch.tensor([[1, 7, 11, 13, 2], [1, 17, 19, 23, 2]], dtype=torch.long)[:batch]
    target = torch.tensor([[1, 29, 31, 2], [1, 37, 41, 2]], dtype=torch.long)[:batch]
    return {
        "input_ids": source,
        "attention_mask": torch.ones_like(source, dtype=torch.bool),
        "source_content_mask": torch.tensor([[False, True, True, True, False]] * batch),
        "decoder_prompt_ids": torch.tensor([[1, 5]] * batch),
        "decoder_prompt_mask": torch.ones(batch, 2, dtype=torch.bool),
        "decoder_input_ids": target,
        "decoder_attention_mask": torch.ones_like(target, dtype=torch.bool),
        "labels": target,
    }


def side_layers(model: SideMemorySummarizer) -> list[DecoderLayerWithCross]:
    return [
        layer
        for layer in model.decoder.backbone.layers
        if isinstance(layer, DecoderLayerWithCross) and layer.side_cross is not None
    ]


def open_side_gates(model: SideMemorySummarizer, value: float = 0.20) -> None:
    with torch.no_grad():
        for layer in side_layers(model):
            assert layer.side_gate is not None
            layer.side_gate.fill_(value)


def test_zero_gate_exactly_matches_direct_projection_at_initialization() -> None:
    full_config = config()
    direct_config = copy.deepcopy(full_config)
    direct_config["architecture"]["bridge_mode"] = "direct_projection"
    torch.manual_seed(11)
    full = SideMemorySummarizer(full_config).eval()
    torch.manual_seed(11)
    direct = SideMemorySummarizer(direct_config).eval()
    with torch.no_grad():
        full_output = full(**inputs()).logits
        direct_output = direct(**inputs()).logits
    torch.testing.assert_close(full_output, direct_output, rtol=0, atol=0)
    assert all(layer.side_gate is not None and layer.side_gate.item() == 0 for layer in side_layers(full))
    assert direct.bridge.resampler is None
    assert not side_layers(direct)


def test_resampler_reads_only_article_content_tokens() -> None:
    model = SideMemorySummarizer(config()).eval()
    batch = inputs()
    observed_masks = []

    def capture_mask(_module, args):
        observed_masks.append(args[1].detach().clone())

    handle = model.bridge.resampler.register_forward_pre_hook(capture_mask)
    try:
        with torch.no_grad():
            model.encode_source(
                batch["input_ids"],
                batch["attention_mask"],
                batch["source_content_mask"],
                batch["decoder_prompt_ids"],
                batch["decoder_prompt_mask"],
                torch.full((2,), 32.0),
            )
    finally:
        handle.remove()
    assert len(observed_masks) == 1
    torch.testing.assert_close(observed_masks[0], batch["source_content_mask"])


def test_gate_learns_first_then_side_parameters_receive_gradients() -> None:
    torch.manual_seed(13)
    model = SideMemorySummarizer(config()).train()
    output = model(**inputs())
    assert output.loss is not None
    output.loss.backward()
    gate_grad = sum(abs(float(layer.side_gate.grad)) for layer in side_layers(model) if layer.side_gate is not None)
    assert gate_grad > 0
    initial_side_grad = sum(
        float(parameter.grad.float().abs().sum())
        for parameter in model.bridge.resampler.parameters()
        if parameter.grad is not None
    )
    assert initial_side_grad == 0

    model.zero_grad(set_to_none=True)
    open_side_gates(model)
    opened = model(**inputs())
    assert opened.loss is not None
    opened.loss.backward()
    opened_side_grad = sum(
        float(parameter.grad.float().abs().sum())
        for parameter in model.bridge.resampler.parameters()
        if parameter.grad is not None
    )
    assert opened_side_grad > 0


def test_project_optimizer_updates_open_side_resampler() -> None:
    torch.manual_seed(17)
    model = SideMemorySummarizer(config()).train()
    open_side_gates(model)
    set_stage_trainability(model, "full_finetune")
    optimizer = build_optimizer(model, model.config, "full_finetune")
    parameter = model.bridge.resampler.o_proj.weight
    before = parameter.detach().clone()
    loss = model(**inputs()).loss
    assert loss is not None
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, parameter.detach())


def test_side_bank_cannot_change_base_memory_or_grounded_copy_state() -> None:
    torch.manual_seed(19)
    model = SideMemorySummarizer(config(copy_enabled=True)).eval()
    batch = inputs()
    copy_inputs = {
        "copy_token_ids": torch.tensor([[43, 47], [53, 59]]),
        "copy_token_mask": torch.ones(2, 2, dtype=torch.bool),
        "copy_encoder_indices": torch.tensor([[1, 2], [1, 2]]),
        "copy_token_indices": torch.tensor([[0, 1], [0, 1]]),
        "copy_alignment_weights": torch.ones(2, 2),
    }
    with torch.no_grad():
        first = model.encode_source(
            batch["input_ids"],
            batch["attention_mask"],
            batch["source_content_mask"],
            batch["decoder_prompt_ids"],
            batch["decoder_prompt_mask"],
            torch.full((2,), 32.0),
            **copy_inputs,
        )
        for parameter in model.bridge.resampler.parameters():
            parameter.add_(torch.randn_like(parameter) * 3)
        second = model.encode_source(
            batch["input_ids"],
            batch["attention_mask"],
            batch["source_content_mask"],
            batch["decoder_prompt_ids"],
            batch["decoder_prompt_mask"],
            torch.full((2,), 32.0),
            **copy_inputs,
        )
    torch.testing.assert_close(first.memory, second.memory, rtol=0, atol=0)
    torch.testing.assert_close(first.source_bias, torch.zeros_like(first.source_bias), rtol=0, atol=0)
    assert first.copy_state is not None and second.copy_state is not None
    for left, right in zip(
        (first.copy_state.keys, first.copy_state.token_ids, first.copy_state.mask, first.copy_state.bias),
        (second.copy_state.keys, second.copy_state.token_ids, second.copy_state.mask, second.copy_state.bias),
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert first.side_memory is not None and second.side_memory is not None
    assert not torch.equal(first.side_memory, second.side_memory)


def test_open_branch_changes_logits_and_zeroing_it_recovers_anchor() -> None:
    torch.manual_seed(23)
    model = SideMemorySummarizer(config()).eval()
    batch = inputs()
    with torch.no_grad():
        anchor = model(**batch).logits
        for parameter in model.bridge.resampler.parameters():
            parameter.add_(0.2 * torch.randn_like(parameter))
        open_side_gates(model, 0.4)
        active = model(**batch).logits
        open_side_gates(model, 0.0)
        recovered = model(**batch).logits
    assert anchor is not None and active is not None and recovered is not None
    assert not torch.allclose(anchor, active)
    torch.testing.assert_close(anchor, recovered, rtol=0, atol=0)


def test_separate_side_kv_cache_matches_uncached_decoder() -> None:
    torch.manual_seed(29)
    model = SideMemorySummarizer(config()).eval()
    open_side_gates(model, 0.25)
    batch = inputs(batch=1)
    with torch.no_grad():
        bridge = model.encode_source(
            batch["input_ids"],
            batch["attention_mask"],
            batch["source_content_mask"],
            batch["decoder_prompt_ids"],
            batch["decoder_prompt_mask"],
            torch.tensor([32.0]),
        )
        full_logits, _, _ = model.decoder(
            batch["decoder_input_ids"],
            bridge.memory,
            bridge.memory_mask,
            bridge.source_bias,
            batch["decoder_attention_mask"],
            side_memory=bridge.side_memory,
            side_memory_mask=bridge.side_memory_mask,
        )
        model.decoder.prepare_cross_cache(bridge.memory, side_memory=bridge.side_memory)
        cached_logits = []
        past = None
        try:
            for position in range(batch["decoder_input_ids"].shape[1]):
                step_logits, past, _ = model.decoder(
                    batch["decoder_input_ids"][:, position : position + 1],
                    bridge.memory,
                    bridge.memory_mask,
                    bridge.source_bias,
                    batch["decoder_attention_mask"][:, : position + 1],
                    past_key_values=past,
                    use_cache=True,
                    side_memory=bridge.side_memory,
                    side_memory_mask=bridge.side_memory_mask,
                )
                assert step_logits is not None
                cached_logits.append(step_logits)
        finally:
            model.decoder.clear_cross_cache()
    assert full_logits is not None
    torch.testing.assert_close(torch.cat(cached_logits, dim=1), full_logits, rtol=2e-5, atol=2e-5)
    for layer in side_layers(model):
        assert layer.side_cross is not None and layer.side_cross._cache is None
