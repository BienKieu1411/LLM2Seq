from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from xov.config import load_config, validate_config
from xov.data.collate import SummarizationCollator
from xov.data.schema import CanonicalRecord
from xov.data.source_alignment import SOURCE_ALIGNMENT_KEYS
from xov.data.source_alignment import align_source_tokens
from xov.modeling.decoder import CopiedCrossAttention
from xov.modeling.grounded_copy import GroundedCopyHead
from xov.modeling.model import XOVModel
from xov.modeling.outputs import EncoderState
from xov.modeling.xov import CrossTokenizerOrderedValueBridge
from xov.evaluation.generate import generate_greedy
from xov.runtime import _TinyTokenizer
from xov.runtime import evaluate


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_programmatic_evaluation_defaults_to_validation() -> None:
    assert inspect.signature(evaluate).parameters["split"].default == "validation"


def _architecture(mode: str = "cross_tokenizer_ordered_value") -> dict:
    return {
        "name": "cross_tokenizer_ordered_value",
        "bridge_mode": mode,
        "lexical_rank": 4,
        "phrase_kernel": 3,
        "phrase_directional": True,
        "value_gate_init": 0.10,
        "output_init_gain": 1.0,
        "value_gate_max": 0.20,
        "residual_reference_rms": 1.0,
        "no_alignment_fallback": "direct_projection",
        "key_memory": "direct_projection",
        "copy_memory": "direct_projection",
    }


def _state(batch: int = 2, source_length: int = 5, hidden: int = 7) -> EncoderState:
    return EncoderState(
        final=torch.randn(batch, source_length, hidden),
        taps=(),
        attention_mask=torch.ones(batch, source_length, dtype=torch.bool),
        content_mask=torch.ones(batch, source_length, dtype=torch.bool),
    )


def _alignment(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "copy_token_positions": torch.arange(3).expand(batch, -1),
        "copy_token_ids": torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=torch.long)[:batch],
        "copy_token_mask": torch.ones(batch, 3, dtype=torch.bool),
        "copy_encoder_indices": torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long)[:batch],
        "copy_token_indices": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)[:batch],
        "copy_alignment_weights": torch.ones(batch, 3),
    }


def _embedding() -> nn.Embedding:
    embedding = nn.Embedding(32, 11)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(32 * 11, dtype=torch.float32).reshape(32, 11) / 100.0)
    return embedding


def test_configs_contain_only_xov_architecture_fields() -> None:
    config = load_config(CONFIG_ROOT / "xov_smoke.yaml")
    validate_config(config)
    assert config["architecture"]["name"] == "cross_tokenizer_ordered_value"
    assert set(config["architecture"]) == {
        "name",
        "bridge_mode",
        "output_init_gain",
        "lexical_rank",
        "phrase_kernel",
        "phrase_directional",
        "value_gate_init",
        "value_gate_max",
        "residual_reference_rms",
        "no_alignment_fallback",
        "key_memory",
        "copy_memory",
    }


def test_initial_keys_and_values_differ_while_copy_preserves_anchor() -> None:
    torch.manual_seed(23)
    xov = CrossTokenizerOrderedValueBridge(7, 11, _architecture())
    torch.manual_seed(23)
    direct = CrossTokenizerOrderedValueBridge(7, 11, _architecture("direct_projection"))
    assert torch.equal(xov.direct_projection.weight, direct.direct_projection.weight)

    state = _state(hidden=7)
    alignment = _alignment()
    xov_out = xov(state, _embedding(), **alignment)
    direct_out = direct(state, _embedding())
    assert torch.equal(xov_out.copy_memory, direct_out.memory)
    assert not torch.equal(xov_out.memory, direct_out.memory)
    assert not torch.equal(xov_out.value_memory, direct_out.memory)
    assert not torch.equal(xov_out.value_memory, xov_out.memory)


def test_reverse_scatter_renormalizes_unequal_many_to_one_overlap() -> None:
    ordered = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    content = torch.ones(1, 2, dtype=torch.bool)
    token_mask = torch.ones(1, 4, dtype=torch.bool)
    encoder_indices = torch.tensor([[0, 1, 1, 0]])
    token_indices = torch.tensor([[0, 1, 2, 3]])
    weights = torch.tensor([[0.25, 0.75, 0.50, 0.50]])
    pooled, aligned = CrossTokenizerOrderedValueBridge._reverse_scatter(
        ordered, content, token_mask, encoder_indices, token_indices, weights
    )
    assert torch.allclose(pooled[0, :, 0], torch.tensor([3.0, 2.4]))
    assert torch.equal(aligned, torch.tensor([[True, True]]))


def test_prefix_padding_and_unaligned_destinations_have_zero_residual() -> None:
    bridge = CrossTokenizerOrderedValueBridge(7, 11, _architecture())
    with torch.no_grad():
        bridge.lexical_up.weight.fill_(0.25)
    state = EncoderState(
        final=torch.randn(1, 4, 7),
        taps=(),
        attention_mask=torch.tensor([[True, True, True, False]]),
        content_mask=torch.tensor([[False, True, True, False]]),
    )
    alignment = {
        "copy_token_positions": torch.arange(3)[None],
        "copy_token_ids": torch.tensor([[4, 5, 6]]),
        "copy_token_mask": torch.tensor([[True, False, True]]),
        "copy_encoder_indices": torch.tensor([[0, 1, 2]]),
        "copy_token_indices": torch.tensor([[0, 1, 2]]),
        "copy_alignment_weights": torch.tensor([[1.0, 1.0, 1.0]]),
    }
    result = bridge(state, _embedding(), **alignment)
    for routed in (result.memory, result.value_memory):
        residual = routed - result.copy_memory
        assert torch.equal(residual[0, 0], torch.zeros_like(residual[0, 0]))
        assert torch.equal(residual[0, 1], torch.zeros_like(residual[0, 1]))
        assert torch.equal(residual[0, 3], torch.zeros_like(residual[0, 3]))
        assert torch.any(residual[0, 2].ne(0))


def test_first_and_second_backward_open_the_lexical_route() -> None:
    torch.manual_seed(4)
    bridge = CrossTokenizerOrderedValueBridge(7, 11, _architecture())
    state = _state(hidden=7)
    embedding = _embedding()
    alignment = _alignment()
    output = bridge(state, embedding, **alignment)
    (output.value_memory.square().mean() + output.memory.square().mean()).backward()
    named = dict(bridge.named_parameters())
    assert named["direct_projection.weight"].grad is not None
    assert named["direct_projection.weight"].grad.isfinite().all()
    assert named["direct_projection.weight"].grad.abs().sum() > 0
    assert named["lexical_up.weight"].grad is not None
    assert named["lexical_up.weight"].grad.isfinite().all()
    assert named["lexical_up.weight"].grad.abs().sum() > 0

    optimizer = torch.optim.SGD(bridge.parameters(), lr=0.1)
    optimizer.step()
    bridge.zero_grad(set_to_none=True)
    output = bridge(state, embedding, **alignment)
    (output.value_memory.square().mean() + output.memory.square().mean()).backward()
    for name in ("lexical_down.weight", "phrase_conv.weight", "value_gate_raw", "key_gate_raw"):
        gradient = named[name].grad
        assert gradient is not None and gradient.isfinite().all() and gradient.abs().sum() > 0, name


def test_empty_alignment_is_identity_and_backward_graph_safe() -> None:
    bridge = CrossTokenizerOrderedValueBridge(7, 11, _architecture())
    state = _state(hidden=7)
    empty = {
        "copy_token_positions": torch.zeros(2, 1, dtype=torch.long),
        "copy_token_ids": torch.zeros(2, 1, dtype=torch.long),
        "copy_token_mask": torch.zeros(2, 1, dtype=torch.bool),
        "copy_encoder_indices": torch.zeros(2, 1, dtype=torch.long),
        "copy_token_indices": torch.zeros(2, 1, dtype=torch.long),
        "copy_alignment_weights": torch.zeros(2, 1),
    }
    output = bridge(state, _embedding(), **empty)
    assert torch.equal(output.value_memory, output.memory)
    assert torch.equal(output.memory, output.copy_memory)
    (output.value_memory.square().mean() + output.memory.square().mean()).backward()
    assert all(parameter.grad is not None and parameter.grad.isfinite().all() for parameter in bridge.parameters())


def test_active_key_and_value_routes_keep_copy_state_on_base_memory() -> None:
    torch.manual_seed(9)
    bridge = CrossTokenizerOrderedValueBridge(7, 11, _architecture())
    with torch.no_grad():
        bridge.lexical_up.weight.fill_(0.1)
    state = _state(hidden=7)
    alignment = _alignment()
    output = bridge(state, _embedding(), **alignment)
    assert torch.any(output.value_memory.ne(output.copy_memory))
    assert torch.any(output.memory.ne(output.copy_memory))
    expected_anchor = bridge.direct_projection(state.final)
    assert torch.equal(output.copy_memory, expected_anchor)
    copy_a = GroundedCopyHead(11, 4, 0.05)
    copy_b = deepcopy(copy_a)
    state_a = copy_a.prepare(
        output.copy_memory,
        output.content_mask,
        _embedding(),
        **{k: v for k, v in alignment.items() if k != "copy_token_positions"},
    )
    state_b = copy_b.prepare(
        expected_anchor,
        output.content_mask,
        _embedding(),
        **{k: v for k, v in alignment.items() if k != "copy_token_positions"},
    )
    assert torch.equal(state_a.keys, state_b.keys)
    assert torch.equal(state_a.token_ids, state_b.token_ids)


def test_cross_attention_cache_keeps_base_keys_and_caches_active_values() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=6,
        max_position_embeddings=64,
    )
    decoder = Qwen3ForCausalLM(config)
    layer = decoder.model.layers[0]
    cross = CopiedCrossAttention(layer.self_attn, layer.input_layernorm, config, 0.0)
    memory = torch.randn(2, 5, 24)
    value_memory = memory + 0.25
    base_key, base_value = cross._memory_kv(memory, memory)
    active_key, active_value = cross._memory_kv(memory, value_memory)
    assert torch.equal(base_key, active_key)
    assert torch.any(base_value.ne(active_value))

    cross.prepare_cache(memory, value_memory)
    assert torch.equal(cross._cache[0], active_key)
    assert torch.equal(cross._cache[1], active_value)
    selected = torch.tensor([1])
    cross._cache = tuple(value.index_select(0, selected) for value in cross._cache)
    assert cross._cache[0].shape[0] == 1
    assert torch.equal(cross._cache[1], active_value.index_select(0, selected))


def test_production_model_ce_steps_open_xov_gradients() -> None:
    config = load_config(CONFIG_ROOT / "xov_smoke.yaml")
    loader = __import__("xov.runtime", fromlist=["build_loaders"]).build_loaders(config, max_train_examples=2)["train"]
    batch = next(iter(loader))
    model = XOVModel(config)
    alignment = {key: batch[key] for key in SOURCE_ALIGNMENT_KEYS}
    forward_args = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "source_content_mask": batch["source_content_mask"],
        "decoder_prompt_ids": batch["decoder_prompt_ids"],
        "decoder_prompt_mask": batch["decoder_prompt_mask"],
        "decoder_input_ids": batch["decoder_input_ids"],
        "decoder_attention_mask": batch["decoder_attention_mask"],
        "labels": batch["labels"],
        "return_logits": False,
        **alignment,
    }
    loss = model(**forward_args).loss_ce
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    named = dict(model.bridge.named_parameters())
    encoder_gradients = [
        parameter.grad
        for parameter in model.encoder.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert encoder_gradients
    assert all(gradient.isfinite().all() for gradient in encoder_gradients)
    assert any(gradient.abs().sum() > 0 for gradient in encoder_gradients)
    assert named["lexical_up.weight"].grad is not None
    assert named["lexical_up.weight"].grad.isfinite().all()
    assert named["lexical_up.weight"].grad.abs().sum() > 0

    for name in (
        "lexical_norm.weight",
        "lexical_down.weight",
        "phrase_conv.weight",
        "lexical_up.weight",
        "value_gate_raw",
        "key_gate_raw",
    ):
        gradient = named[name].grad
        assert gradient is not None and gradient.isfinite().all() and gradient.abs().sum() > 0, name
    before = {name: value.detach().clone() for name, value in named.items()}
    optimizer = torch.optim.SGD(model.bridge.parameters(), lr=0.1)
    optimizer.step()
    for name in (
        "lexical_norm.weight",
        "lexical_down.weight",
        "phrase_conv.weight",
        "lexical_up.weight",
        "value_gate_raw",
        "key_gate_raw",
    ):
        assert not torch.equal(before[name], named[name]), name
    model.zero_grad(set_to_none=True)
    second_loss = model(**forward_args).loss_ce
    assert second_loss is not None and torch.isfinite(second_loss)
    second_loss.backward()
    for name in ("lexical_down.weight", "phrase_conv.weight", "value_gate_raw", "key_gate_raw"):
        gradient = named[name].grad
        assert gradient is not None and gradient.isfinite().all() and gradient.abs().sum() > 0, name


def test_production_decoder_logits_change_when_key_or_value_route_is_active() -> None:
    config = load_config(CONFIG_ROOT / "xov_smoke.yaml")
    loader = __import__("xov.runtime", fromlist=["build_loaders"]).build_loaders(config, max_train_examples=2)["train"]
    batch = next(iter(loader))
    model = XOVModel(config)
    model.eval()
    with torch.no_grad():
        model.bridge.lexical_up.weight.fill_(0.1)
        bridge = model.encode_source(
            batch["input_ids"],
            batch["attention_mask"],
            batch["source_content_mask"],
            batch["decoder_prompt_ids"],
            batch["decoder_prompt_mask"],
            torch.full((batch["input_ids"].shape[0],), 4.0),
            **{key: batch[key] for key in SOURCE_ALIGNMENT_KEYS},
        )
        assert torch.any(bridge.value_memory.ne(bridge.memory))
        active_logits, _, _ = model.decoder(
            batch["decoder_input_ids"],
            bridge.memory,
            bridge.memory_mask,
            batch["decoder_attention_mask"],
            return_logits=True,
            value_memory=bridge.value_memory,
        )
        value_only_logits, _, _ = model.decoder(
            batch["decoder_input_ids"],
            bridge.copy_memory,
            bridge.memory_mask,
            batch["decoder_attention_mask"],
            return_logits=True,
            value_memory=bridge.value_memory,
        )
        key_only_logits, _, _ = model.decoder(
            batch["decoder_input_ids"],
            bridge.memory,
            bridge.memory_mask,
            batch["decoder_attention_mask"],
            return_logits=True,
            value_memory=bridge.copy_memory,
        )
    assert active_logits is not None and value_only_logits is not None and key_only_logits is not None
    assert torch.any(active_logits.ne(value_only_logits))
    assert torch.any(active_logits.ne(key_only_logits))


def test_production_cached_generation_compacts_active_value_state() -> None:
    config = load_config(CONFIG_ROOT / "xov_smoke.yaml")
    from xov.runtime import build_loaders

    loader = build_loaders(config, split="test", batch_size_override=2)["test"]
    loader.collate_fn.include_targets = False
    batch = next(iter(loader))
    model = XOVModel(config)
    model.eval()
    with torch.no_grad():
        model.bridge.lexical_up.weight.fill_(0.1)
        expected_bridge = model.encode_source(
            batch["input_ids"],
            batch["attention_mask"],
            batch["source_content_mask"],
            batch["decoder_prompt_ids"],
            batch["decoder_prompt_mask"],
            torch.full((batch["input_ids"].shape[0],), 2.0),
            **{key: batch[key] for key in SOURCE_ALIGNMENT_KEYS},
        )
        reorder = torch.tensor([1, 1, 0])
        reordered_bridge = expected_bridge.index_select(reorder)
        assert torch.equal(reordered_bridge.value_memory, expected_bridge.value_memory.index_select(0, reorder))
        assert torch.equal(reordered_bridge.copy_memory, expected_bridge.copy_memory.index_select(0, reorder))

    class ControlledHead(nn.Module):
        def __init__(self, original: nn.Module, eos_token: int, survivor_token: int):
            super().__init__()
            self.original = original
            self.eos_token = eos_token
            self.survivor_token = survivor_token

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            logits = self.original(hidden_states)
            controlled = torch.full_like(logits, -100.0)
            if logits.shape[0] == 2:
                controlled[0, :, self.eos_token] = 100.0
                controlled[1, :, self.survivor_token] = 100.0
            else:
                controlled[:, :, self.survivor_token] = 100.0
            return controlled

    model.decoder.lm_head = ControlledHead(model.decoder.lm_head, eos_token=2, survivor_token=4)
    selected_indices: list[torch.Tensor] = []
    cache_snapshots: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    original_select = model.decoder.select_cross_cache

    def observe_select(indices: torch.Tensor) -> None:
        original_select(indices)
        selected_indices.append(indices.detach().clone())
        cache_snapshots.append(
            [
                (layer.cross._cache[0].clone(), layer.cross._cache[1].clone())
                for layer in model.decoder.backbone.layers
                if layer.cross._cache is not None
            ]
        )

    model.decoder.select_cross_cache = observe_select
    texts, _ = generate_greedy(
        model,
        batch,
        loader.collate_fn.decoder_tokenizer,
        max_new_tokens=2,
        min_new_tokens=0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        compact_finished=True,
    )
    assert texts[0] == ""
    assert texts[1] == "tok4 tok4"
    assert len(selected_indices) == 1
    assert torch.equal(selected_indices[0], torch.tensor([1]))
    assert len(cache_snapshots) == 1
    expected_active = expected_bridge.index_select(torch.tensor([1]))
    expected_caches = []
    for layer in model.decoder.backbone.layers:
        key, value = layer.cross._memory_kv(expected_active.memory, expected_active.value_memory)
        expected_caches.append((key, value))
    for (actual_key, actual_value), (expected_key, expected_value) in zip(cache_snapshots[0], expected_caches):
        torch.testing.assert_close(actual_key, expected_key, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-6, atol=1e-6)


def test_collator_alignment_rules_for_xov_and_direct_modes() -> None:
    encoder = _TinyTokenizer()
    decoder = _TinyTokenizer()
    data = {
        "source_field": "text",
        "target_field": "summary",
        "id_field": "id",
        "list_separator": "\n",
        "encoder_prefix": "",
        "decoder_prompt": "Summarize:",
        "decoder_prefix": "",
        "decoder_chat_template": False,
        "max_source_length": 64,
        "max_target_length": 16,
    }
    record = CanonicalRecord("row", "Cats eat fish.", "Cats eat fish.")
    xov = SummarizationCollator(encoder, decoder, data, grounded_copy=False, source_alignment=True)([record])
    assert set(SOURCE_ALIGNMENT_KEYS).issubset(xov)

    direct = SummarizationCollator(encoder, decoder, data, grounded_copy=False, source_alignment=False)([record])
    assert not any(key in direct for key in SOURCE_ALIGNMENT_KEYS)

    direct_copy = SummarizationCollator(encoder, decoder, data, grounded_copy=True, source_alignment=False)([record])
    assert set(SOURCE_ALIGNMENT_KEYS).issubset(direct_copy)


def test_truncated_partial_final_decoder_token_is_excluded() -> None:
    source = "Spain 2003 hidden"

    class BoundaryTokenizer:
        all_special_ids = ()
        pad_token_id = bos_token_id = eos_token_id = unk_token_id = None

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            if len(text) == 8:
                result = {"input_ids": [10, 11], "offset_mapping": [(0, 5), (6, 8)]}
            else:
                result = {
                    "input_ids": [10, 11, 12],
                    "offset_mapping": [(0, 5), (6, 10), (11, 17)],
                }
            return result

    tokenizer = BoundaryTokenizer()
    truncated = align_source_tokens(source, 0, [(0, 5), (6, 8)], tokenizer)
    full = align_source_tokens(source, 0, [(0, 5), (6, 10), (11, 17)], tokenizer)
    assert truncated["copy_token_ids"] == [10]
    assert full["copy_token_ids"] == [10, 11, 12]


def test_direct_projection_without_copy_does_not_require_alignment() -> None:
    config = load_config(CONFIG_ROOT / "xov_smoke.yaml")
    config["architecture"]["bridge_mode"] = "direct_projection"
    config["decoder"]["grounded_copy"]["enabled"] = False
    model_config = deepcopy(config)
    from xov.modeling.model import XOVModel
    from xov.runtime import build_loaders

    loader = build_loaders(model_config, split="test", batch_size_override=2)["test"]
    loader.collate_fn.include_targets = False
    batch = next(iter(loader))
    model = XOVModel(model_config)
    output = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["source_content_mask"],
        batch["decoder_prompt_ids"],
        batch["decoder_prompt_mask"],
        batch["decoder_input_ids"],
        batch["decoder_attention_mask"],
        return_logits=False,
    )
    assert output.bridge.value_memory is output.bridge.memory
