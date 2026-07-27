from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from llm2seq_v5.decoder import PretrainedQwenDecoder
from llm2seq_v5.generation import generate


class _RecordingBackbone(nn.Module):
    """Small HF-backbone stand-in; it never initializes a real checkpoint."""

    def __init__(self, vocab_size: int = 23, hidden_size: int = 6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            values = torch.arange(vocab_size * hidden_size, dtype=torch.float32)
            self.embed_tokens.weight.copy_(values.view(vocab_size, hidden_size) / 100.0)
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise AssertionError("exactly one decoder input representation is required")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        self.calls.append(
            {
                "input_ids": None if input_ids is None else input_ids.detach().clone(),
                "inputs_embeds": (None if inputs_embeds is None else inputs_embeds.detach().clone()),
                "attention_mask": (None if attention_mask is None else attention_mask.detach().clone()),
                "past_key_values": past_key_values,
            }
        )
        cache = ("synthetic-cache", len(self.calls)) if use_cache else None
        return SimpleNamespace(last_hidden_state=hidden + 7.0, past_key_values=cache)


def _synthetic_decoder() -> tuple[PretrainedQwenDecoder, _RecordingBackbone]:
    decoder = PretrainedQwenDecoder.__new__(PretrainedQwenDecoder)
    nn.Module.__init__(decoder)
    backbone = _RecordingBackbone()
    decoder.backbone = backbone
    return decoder, backbone


def test_prefix_is_prepended_once_and_hidden_states_remain_label_aligned():
    decoder, backbone = _synthetic_decoder()
    input_ids = torch.tensor([[2, 3, 4], [5, 6, 7]])
    token_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    prefix = torch.randn(2, 2, 6, dtype=torch.float64)
    prefix_mask = torch.tensor([[1, 0], [1, 1]])

    states, cache = decoder(
        input_ids=input_ids,
        attention_mask=token_mask,
        encoder_hidden_states=torch.randn(2, 4, 6),
        summary_prefix=prefix,
        summary_prefix_mask=prefix_mask,
        use_cache=True,
    )

    # Prefix positions were present inside the backbone but are absent from
    # the returned tensor consumed by the LM head and teacher-forcing labels.
    assert states.shape == (2, 3, 6)
    assert torch.allclose(states, backbone.embed_tokens(input_ids) + 7.0)
    assert cache == ("synthetic-cache", 1)
    call = backbone.calls[0]
    assert call["input_ids"] is None
    assert call["inputs_embeds"].shape == (2, 5, 6)
    assert call["inputs_embeds"].dtype == backbone.embed_tokens.weight.dtype
    assert torch.allclose(call["inputs_embeds"][:, :2], prefix.float())
    assert torch.equal(
        call["attention_mask"],
        torch.cat([prefix_mask, token_mask], dim=1),
    )


def test_prefix_can_match_native_embedding_rms_at_injection_boundary():
    decoder, backbone = _synthetic_decoder()
    decoder.summary_prefix_scale_mode = "match_embedding_rms"
    decoder.summary_prefix_scale_multiplier = 1.0
    decoder.summary_prefix_scale_epsilon = 1e-6
    input_ids = torch.tensor([[2, 3, 4], [5, 6, 7]])
    token_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    prefix = torch.full((2, 2, 6), 20.0)
    prefix_before = prefix.clone()
    prefix_mask = torch.tensor([[1, 0], [1, 1]])

    decoder(
        input_ids=input_ids,
        attention_mask=token_mask,
        encoder_hidden_states=torch.randn(2, 4, 6),
        summary_prefix=prefix,
        summary_prefix_mask=prefix_mask,
    )

    injected = backbone.calls[0]["inputs_embeds"][:, :2]
    expected_rms = decoder._masked_rms(backbone.embed_tokens(input_ids), token_mask)
    injected_rms = decoder._masked_rms(injected, prefix_mask)
    logged_prefix_rms, logged_embedding_rms, logged_ratio = decoder.prefix_input_scale_metrics()
    assert torch.allclose(injected_rms, expected_rms, rtol=1e-5, atol=1e-7)
    assert torch.allclose(logged_prefix_rms, injected_rms)
    assert torch.allclose(logged_embedding_rms, expected_rms)
    assert torch.allclose(logged_ratio, torch.ones_like(logged_ratio), rtol=1e-5)
    assert torch.equal(prefix, prefix_before)


def test_cached_step_keeps_prefix_only_in_kv_cache_and_does_not_prepend_again():
    decoder, backbone = _synthetic_decoder()
    prefix = torch.randn(1, 2, 6)
    prefix_mask = torch.tensor([[1, 0]])
    _, past = decoder(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        encoder_hidden_states=torch.randn(1, 4, 6),
        summary_prefix=prefix,
        summary_prefix_mask=prefix_mask,
        use_cache=True,
    )
    full_cached_mask = torch.tensor([[1, 0, 1, 1, 1, 1]])
    states, next_past = decoder(
        input_ids=torch.tensor([[4]]),
        attention_mask=full_cached_mask,
        encoder_hidden_states=torch.randn(1, 4, 6),
        summary_prefix=prefix,
        summary_prefix_mask=prefix_mask,
        past_key_values=past,
        use_cache=True,
    )

    assert states.shape == (1, 1, 6)
    assert next_past == ("synthetic-cache", 2)
    assert backbone.calls[1]["inputs_embeds"] is None
    assert torch.equal(backbone.calls[1]["input_ids"], torch.tensor([[4]]))
    assert torch.equal(backbone.calls[1]["attention_mask"], full_cached_mask)


def test_no_prefix_keeps_the_original_input_ids_path():
    decoder, backbone = _synthetic_decoder()
    input_ids = torch.tensor([[8, 9]])
    mask = torch.ones_like(input_ids)
    states, cache = decoder(
        input_ids=input_ids,
        attention_mask=mask,
        encoder_hidden_states=torch.randn(1, 4, 6),
        use_cache=False,
    )
    assert cache is None
    assert torch.allclose(states, backbone.embed_tokens(input_ids) + 7.0)
    assert torch.equal(backbone.calls[0]["input_ids"], input_ids)
    assert backbone.calls[0]["inputs_embeds"] is None
    assert torch.equal(backbone.calls[0]["attention_mask"], mask)


@pytest.mark.parametrize(
    ("prefix", "prefix_mask", "message"),
    [
        (torch.randn(2, 6), None, "summary_prefix must be"),
        (torch.randn(2, 2, 5), None, "hidden size"),
        (torch.randn(3, 2, 6), None, "batch"),
        (torch.randn(2, 2, 6), torch.ones(2, 3), "summary_prefix_mask"),
    ],
)
def test_prefix_shape_validation(prefix, prefix_mask, message):
    decoder, _ = _synthetic_decoder()
    with pytest.raises(ValueError, match=message):
        decoder(
            input_ids=torch.ones(2, 3, dtype=torch.long),
            encoder_hidden_states=torch.randn(2, 4, 6),
            summary_prefix=prefix,
            summary_prefix_mask=prefix_mask,
        )


class _GenerationDecoder:
    def __init__(self):
        self.calls: list[dict[str, object]] = []
        self.cleared = False

    def prepare_cross_attention_cache(self, memory):
        self.cached_memory = memory

    def clear_cross_attention_cache(self):
        self.cleared = True

    def memory_routing_per_layer(self):
        return torch.ones(1, 1)

    def set_generation_routing_statistics(self, per_layer, observations):
        self.routing_statistics = (per_layer, observations)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch, length = kwargs["input_ids"].shape
        states = torch.zeros(batch, length, 4)
        return states, ("past", len(self.calls))


class _GenerationModel:
    def __init__(self, adapter_output):
        self.adapter_output = adapter_output
        self.decoder = _GenerationDecoder()

    def encode(self, input_ids, attention_mask, unit_ids=None):
        return self.adapter_output

    def lm_head(self, states):
        return torch.zeros(states.shape[0], 11)


def test_generation_passes_prefix_every_call_but_builds_cached_mask_once():
    prefix = torch.randn(1, 2, 4)
    prefix_mask = torch.tensor([[1, 0]])
    adapter_output = SimpleNamespace(
        memory=torch.randn(1, 3, 4),
        memory_mask=torch.ones(1, 3, dtype=torch.long),
        attention_bias=None,
        summary_prefix=prefix,
        summary_prefix_mask=prefix_mask,
    )
    model = _GenerationModel(adapter_output)
    output = generate(
        model,
        input_ids=torch.tensor([[4, 5, 6]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        decoder_seed=[1, 2],
        max_new_tokens=2,
        eos_token_id=None,
    )

    assert output.shape == (1, 2)
    assert len(model.decoder.calls) == 2
    first, cached = model.decoder.calls
    assert first["summary_prefix"] is prefix
    assert cached["summary_prefix"] is prefix
    assert first["summary_prefix_mask"] is prefix_mask
    assert cached["summary_prefix_mask"] is prefix_mask
    assert first["past_key_values"] is None
    assert cached["past_key_values"] is not None
    assert torch.equal(first["attention_mask"], torch.ones(1, 2, dtype=torch.long))
    assert torch.equal(
        cached["attention_mask"],
        torch.tensor([[1, 0, 1, 1, 1]]),
    )
    assert model.decoder.cleared is True
