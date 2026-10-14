from __future__ import annotations

import pytest
import torch


def _tiny_causal_lm(family: str):
    from transformers import (
        GemmaConfig,
        GemmaForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    common = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "attention_dropout": 0.0,
    }
    configs = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "mistral": (MistralConfig, MistralForCausalLM),
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "gemma": (GemmaConfig, GemmaForCausalLM),
    }
    config_class, model_class = configs[family]
    config_kwargs = dict(common)
    if family in {"qwen3", "gemma"}:
        config_kwargs["head_dim"] = 8
    config = config_class(**config_kwargs)
    return model_class(config), config


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2", "qwen3", "gemma"])
def test_common_causal_decoder_families_run_and_backpropagate(monkeypatch, family):
    from eviseq_afmr.modeling import decoder as decoder_module

    base_model, model_config = _tiny_causal_lm(family)
    monkeypatch.setattr(decoder_module, "_load_decoder", lambda *args, **kwargs: (base_model, model_config))
    decoder = decoder_module.CausalLMWithCrossAttention(
        f"local-{family}",
        {"cross_attention_every": 1, "cross_gate_init": 0.1, "cross_gate_max": 0.5},
        torch.float32,
        gradient_checkpointing=False,
    )

    tokens = torch.randint(0, model_config.vocab_size, (2, 5))
    memory = torch.randn(2, 7, model_config.hidden_size, requires_grad=True)
    memory_mask = torch.ones(2, 7, dtype=torch.bool)
    source_bias = torch.zeros(2, 7, requires_grad=True)
    attention_mask = torch.ones_like(tokens, dtype=torch.bool)
    logits, _, loss = decoder(
        tokens,
        memory,
        memory_mask,
        source_bias,
        attention_mask=attention_mask,
        labels=tokens,
    )

    assert logits.shape == (2, 5, model_config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert memory.grad is not None and torch.isfinite(memory.grad).all() and memory.grad.abs().sum() > 0
    assert source_bias.grad is not None and torch.isfinite(source_bias.grad).all()


def test_llama_cross_attention_cache_matches_full_decode(monkeypatch):
    from eviseq_afmr.modeling import decoder as decoder_module

    base_model, model_config = _tiny_causal_lm("llama")
    monkeypatch.setattr(decoder_module, "_load_decoder", lambda *args, **kwargs: (base_model, model_config))
    decoder = decoder_module.CausalLMWithCrossAttention(
        "local-llama",
        {"cross_attention_every": 1, "cross_gate_init": 0.1, "cross_gate_max": 0.5},
        torch.float32,
        gradient_checkpointing=False,
    ).eval()
    tokens = torch.tensor([[1, 2, 3]])
    memory = torch.randn(1, 6, model_config.hidden_size)
    memory_mask = torch.ones(1, 6, dtype=torch.bool)
    source_bias = torch.zeros(1, 6)

    with torch.no_grad():
        decoder.prepare_cross_cache(memory)
        _, cache, _ = decoder(
            tokens[:, :2], memory, memory_mask, source_bias, attention_mask=torch.ones(1, 2), use_cache=True
        )
        cached, _, _ = decoder(
            tokens[:, 2:],
            memory,
            memory_mask,
            source_bias,
            attention_mask=torch.ones(1, 3),
            past_key_values=cache,
            use_cache=True,
        )
        decoder.clear_cross_cache()
        full, _, _ = decoder(tokens, memory, memory_mask, source_bias, attention_mask=torch.ones_like(tokens))

    torch.testing.assert_close(cached[:, -1], full[:, -1], atol=1e-5, rtol=1e-5)


def test_gemma2_softcapping_is_rejected_explicitly(monkeypatch):
    from transformers import Gemma2Config, Gemma2ForCausalLM

    from eviseq_afmr.modeling import decoder as decoder_module

    config = Gemma2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = Gemma2ForCausalLM(config)
    monkeypatch.setattr(decoder_module, "_load_decoder", lambda *args, **kwargs: (model, config))

    with pytest.raises(ValueError, match="softcapping"):
        decoder_module.CausalLMWithCrossAttention(
            "local-gemma2",
            {"cross_attention_every": 1},
            torch.float32,
            gradient_checkpointing=False,
        )
