import tempfile
from pathlib import Path

import pytest
import torch
from genbridge.generation import (
    OutputOnlyNoRepeatNGramLogitsProcessor,
    OutputOnlyRepetitionPenaltyLogitsProcessor,
    autoregressive_generate,
)
from genbridge.model import GenBridgeSeq2Seq
from genbridge.pretrained_decoder import CrossAttentionInjectedLayer, QwenCrossAttention
from genbridge.source_encoder import CausalSourceEncoder
from transformers import Qwen3Config, Qwen3ForCausalLM


class _FixedLogitDecoder:
    def __call__(self, input_ids, **kwargs):
        states = torch.zeros((*input_ids.shape, 1), device=input_ids.device)
        return states, None


class _FixedLogitModel:
    def __init__(self):
        self.decoder = _FixedLogitDecoder()

    def eval(self):
        return self

    def encode(self, input_ids, attention_mask, **kwargs):
        return None

    def decoder_memory_kwargs(self, bridge_output):
        return {}

    def lm_head(self, states):
        logits = torch.zeros((states.shape[0], 8), device=states.device)
        logits[:, 3] = 10.0
        logits[:, 4] = 6.0
        return logits


def test_generation_penalties_ignore_fixed_decoder_prompt():
    model = _FixedLogitModel()
    source = torch.tensor([[1, 2]])
    source_mask = torch.ones_like(source)
    repeated_prompt_token = autoregressive_generate(
        model,
        source,
        source_mask,
        max_new_tokens=1,
        decoder_prefix_ids=[3],
        repetition_penalty=2.0,
        use_cache=False,
    )
    unigram_from_prompt = autoregressive_generate(
        model,
        source,
        source_mask,
        max_new_tokens=1,
        decoder_prefix_ids=[3],
        no_repeat_ngram_size=1,
        use_cache=False,
    )
    assert repeated_prompt_token.tolist() == [[3]]
    assert unigram_from_prompt.tolist() == [[3]]


def test_generation_enforces_exact_minimum_number_of_new_tokens():
    model = _FixedLogitModel()
    source = torch.tensor([[1, 2]])
    generated = autoregressive_generate(
        model,
        source,
        torch.ones_like(source),
        max_new_tokens=4,
        min_new_tokens=2,
        eos_token_id=3,
        pad_token_id=0,
        decoder_prefix_ids=[7],
        use_cache=False,
    )
    # EOS is the model's preferred token. It must remain blocked for two full
    # output positions and becomes legal only on the third prediction.
    assert generated.tolist() == [[4, 4, 3]]


def test_direct_generation_processors_ignore_source_prompt():
    scores = torch.zeros((1, 8))
    scores[:, 3] = 10.0
    scores[:, 4] = 6.0
    repetition = OutputOnlyRepetitionPenaltyLogitsProcessor(2.0, prompt_length=1)
    no_unigram = OutputOnlyNoRepeatNGramLogitsProcessor(1, prompt_length=1)

    prompt_only = torch.tensor([[3]])
    torch.testing.assert_close(repetition(prompt_only, scores.clone()), scores)
    assert torch.isfinite(no_unigram(prompt_only, scores.clone())[0, 3])

    one_generated_token = torch.tensor([[3, 3]])
    penalized = repetition(one_generated_token, scores.clone())
    blocked = no_unigram(one_generated_token, scores.clone())
    assert penalized[0, 3].item() == 5.0
    assert torch.isneginf(blocked[0, 3])


def test_source_states_are_invariant_to_left_padding():
    text_config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        use_cache=False,
    )
    with tempfile.TemporaryDirectory() as checkpoint:
        Qwen3ForCausalLM(text_config).save_pretrained(checkpoint)
        encoder = CausalSourceEncoder(
            {
                "encoder_name": checkpoint,
                "dtype": "float32",
                "train_base": True,
                "gradient_checkpointing": False,
                "num_summary_tokens": 2,
                "layer_fusion": {"enabled": False},
            }
        ).eval()
        unpadded = encoder(
            torch.tensor([[3, 4, 5]]),
            torch.tensor([[1, 1, 1]]),
        )
        padded = encoder(
            torch.tensor([[0, 0, 3, 4, 5]]),
            torch.tensor([[0, 0, 1, 1, 1]]),
        )
        torch.testing.assert_close(
            unpadded.token_states,
            padded.token_states[:, -3:],
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            unpadded.plan_states,
            padded.plan_states,
            atol=1e-5,
            rtol=1e-5,
        )


def test_cross_attention_masks_padding_and_cache_preserves_outputs():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    torch.manual_seed(4)
    attention = QwenCrossAttention(config).eval()
    query = torch.randn(2, 3, 32)
    memory = torch.randn(2, 4, 32)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    changed_padding = memory.clone()
    changed_padding[0, 3] = 1000.0
    changed_padding[1, 2:] = -1000.0

    expected = attention(query, memory, mask)
    changed = attention(query, changed_padding, mask)
    torch.testing.assert_close(expected, changed, atol=1e-5, rtol=1e-5)

    attention.prepare_memory_cache(memory)
    cached = attention(query, memory, mask)
    torch.testing.assert_close(expected, cached, atol=1e-5, rtol=1e-5)


def test_injected_cross_attention_reuses_native_self_attention_norms():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    native = Qwen3ForCausalLM(config).model.layers[0]
    with torch.no_grad():
        native.input_layernorm.weight.copy_(torch.linspace(0.5, 2.0, 32))
        native.post_attention_layernorm.weight.fill_(7.0)
    injected = CrossAttentionInjectedLayer(
        native,
        config,
        native.self_attn,
        dropout=0.0,
        memory_attention="gated_dual",
        plan_gate_init=0.1,
        cross_gate_init=0.01,
    )
    torch.testing.assert_close(
        injected.cross_attn_norm.weight,
        native.input_layernorm.weight,
    )
    torch.testing.assert_close(
        injected.cross_attn.memory_norm.weight,
        native.input_layernorm.weight,
    )
    assert not torch.equal(
        injected.cross_attn_norm.weight,
        native.post_attention_layernorm.weight,
    )


def test_mismatched_encoder_decoder_checkpoint_is_rejected_before_loading():
    config = {
        "model": {"encoder_name": "Qwen/Qwen3-Embedding-0.6B"},
        "decoder": {"pretrained_name": "Qwen/Qwen3-0.6B"},
    }
    with pytest.raises(ValueError, match="allow_mixed_checkpoints=true"):
        GenBridgeSeq2Seq(config, vocab_size=128)


def test_mixed_qwen3_hidden_sizes_forward_backward():
    encoder_config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        use_cache=False,
        tie_word_embeddings=True,
    )
    decoder_config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        use_cache=False,
        tie_word_embeddings=True,
    )
    with tempfile.TemporaryDirectory() as directory:
        encoder_checkpoint = Path(directory) / "encoder"
        decoder_checkpoint = Path(directory) / "decoder"
        Qwen3ForCausalLM(encoder_config).save_pretrained(encoder_checkpoint)
        Qwen3ForCausalLM(decoder_config).save_pretrained(decoder_checkpoint)
        config = {
            "model": {
                "encoder_name": str(encoder_checkpoint),
                "allow_mixed_checkpoints": True,
                "dtype": "float32",
                "gradient_checkpointing": False,
                "num_summary_tokens": 2,
                "layer_fusion": {"enabled": False},
            },
            "bridge": {
                "mode": "genbridge",
                "hidden_size": 32,
                "token_num_layers": 1,
                "unit_num_layers": 1,
                "num_heads": 4,
                "ffn_size": 64,
                "dropout": 0.0,
            },
            "decoder": {
                "pretrained_name": str(decoder_checkpoint),
                "num_layers": 2,
                "gradient_checkpointing": False,
                "cross_attention_every": 2,
                "memory_attention": "gated_dual",
            },
            "objectives": {
                "salience_weight": 0.2,
                "plan_alignment_weight": 0.1,
                "plan_diversity_weight": 0.01,
            },
        }
        model = GenBridgeSeq2Seq(config, vocab_size=128)
        assert model.encoder.hidden_size == 64
        assert model.bridge.encoder_size == 64
        assert model.bridge.decoder_size == 32
        assert model.decoder.hidden_size == 32

        source = torch.randint(3, 128, (2, 8))
        target = torch.randint(3, 128, (2, 5))
        output = model(
            input_ids=source,
            attention_mask=torch.ones_like(source),
            unit_ids=torch.tensor([[0, 1, 1, 1, 2, 2, 2, 0]]).repeat(2, 1),
            evidence_labels=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            decoder_input_ids=target,
            decoder_attention_mask=torch.ones_like(target),
            labels=target,
        )
        output["loss"].backward()
        assert output["logits"].shape == (10, 128)
        assert torch.isfinite(output["loss"])


def test_tiny_qwen3_forward_backward_and_cached_decode():
    text_config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        use_cache=False,
        tie_word_embeddings=True,
    )
    with tempfile.TemporaryDirectory() as checkpoint:
        Qwen3ForCausalLM(text_config).save_pretrained(checkpoint)
        config = {
            "model": {
                "encoder_name": checkpoint,
                "dtype": "float32",
                "train_base": True,
                "gradient_checkpointing": False,
                "num_summary_tokens": 2,
                "layer_fusion": {"enabled": False},
            },
            "bridge": {
                "mode": "genbridge",
                "hidden_size": 32,
                "token_num_layers": 1,
                "unit_num_layers": 1,
                "num_heads": 4,
                "ffn_size": 64,
                "dropout": 0.0,
            },
            "decoder": {
                "pretrained_name": checkpoint,
                "num_layers": 4,
                "gradient_checkpointing": False,
                "cross_attention_every": 4,
                "memory_attention": "gated_dual",
                "plan_gate_init": 0.1,
            },
            "objectives": {
                "salience_weight": 0.2,
                "plan_alignment_weight": 0.1,
                "plan_diversity_weight": 0.01,
            },
        }
        model = GenBridgeSeq2Seq(config, text_config.vocab_size)
        model.set_training_stage("interface_warmup")
        assert not next(model.encoder.model.parameters()).requires_grad
        assert model.encoder.summary_tokens.requires_grad
        model.set_training_stage("full_finetune")
        assert all(parameter.requires_grad for parameter in model.parameters())
        source = torch.randint(0, 128, (2, 10))
        source_mask = torch.ones_like(source)
        unit_ids = torch.tensor([[0, 1, 1, 1, 2, 2, 2, 3, 3, 0]]).repeat(2, 1)
        target = torch.randint(0, 128, (2, 6))
        labels = target.clone()
        labels[0, -2:] = -100
        output = model(
            input_ids=source,
            attention_mask=source_mask,
            unit_ids=unit_ids,
            evidence_labels=torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            decoder_input_ids=target,
            decoder_attention_mask=torch.ones_like(target),
            labels=labels,
        )
        output["loss"].backward()
        assert output["logits"].shape == (10, 128)
        assert torch.isfinite(output["loss"])
        assert torch.isfinite(output["cross_gate_mean"])
        assert output["cross_gate_mean"] > 0
        assert output["cross_residual_ratio"] > 0
        assert output["plan_gate_mean"] > 0
        assert output["token_adapter_gate"] > 0
        assert output["plan_adapter_gate"] > 0
        assert "salience_predicted_positive_rate" in output
        assert "salience_precision" in output
        assert "salience_recall" in output
        assert model.encoder.summary_tokens.grad is not None
        injected = model.decoder.backbone.layers[-1]
        assert injected.plan_gate.weight.grad is not None
        assert model.decoder.plan_gate_means()
        assert abs(model.decoder.plan_gate_means()[0] - 0.1) < 0.05

        model.eval()
        bridge_output = model.encode(source, source_mask, unit_ids, return_bridge_output=True)
        memory_kwargs = model.decoder_memory_kwargs(bridge_output)
        model.decoder.prepare_cross_attention_cache(
            memory_kwargs["encoder_hidden_states"],
            token_encoder_hidden_states=memory_kwargs["token_encoder_hidden_states"],
            plan_encoder_hidden_states=memory_kwargs["plan_encoder_hidden_states"],
        )
        first, cache = model.decoder(
            input_ids=target[:, :1],
            use_cache=True,
            **memory_kwargs,
        )
        second, cache = model.decoder(
            input_ids=target[:, 1:2],
            past_key_values=cache,
            use_cache=True,
            **memory_kwargs,
        )
        assert first.shape == second.shape == (2, 1, 64)
        assert cache is not None
        assert len(injected.cross_attn._memory_cache) == 2
        model.decoder.clear_cross_attention_cache()
        assert not injected.cross_attn._memory_cache

        generated, generation_diagnostics = autoregressive_generate(
            model,
            source,
            source_mask,
            unit_ids=unit_ids,
            max_new_tokens=2,
            eos_token_id=None,
            pad_token_id=0,
            decoder_prefix_ids=[1, 17, 18],
            use_cache=True,
            return_diagnostics=True,
        )
        # The fixed task prefix conditions generation but is excluded from the
        # returned predictions and therefore from ROUGE.
        assert generated.shape == (2, 2)
        assert len(generation_diagnostics["plan_gate_layer_sums"]) == 1
        assert len(generation_diagnostics["plan_gate_step_sums"]) == 2
        assert all(
            count > 0 for count in generation_diagnostics["plan_gate_step_counts"]
        )


def test_tiny_qwen3_forward_backward_uses_gated_dual_memory():
    text_config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        use_cache=False,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    with tempfile.TemporaryDirectory() as checkpoint:
        Qwen3ForCausalLM(text_config).save_pretrained(checkpoint)
        config = {
            "model": {
                "encoder_name": checkpoint,
                "dtype": "float32",
                "train_base": True,
                "gradient_checkpointing": False,
                "num_summary_tokens": 2,
                "layer_fusion": {"enabled": False},
            },
            "bridge": {
                "mode": "genbridge",
                "hidden_size": 32,
                "token_num_layers": 1,
                "unit_num_layers": 1,
                "num_heads": 4,
                "ffn_size": 64,
                "dropout": 0.0,
            },
            "decoder": {
                "pretrained_name": checkpoint,
                "num_layers": 4,
                "gradient_checkpointing": False,
                "cross_attention_every": 4,
                "memory_attention": "gated_dual",
                "plan_gate_init": 0.1,
            },
            "objectives": {
                "salience_weight": 0.2,
                "plan_alignment_weight": 0.1,
                "plan_diversity_weight": 0.01,
            },
        }
        model = GenBridgeSeq2Seq(config, text_config.vocab_size)
        source = torch.randint(3, 128, (2, 8))
        source_mask = torch.ones_like(source)
        unit_ids = torch.tensor([[0, 1, 1, 1, 2, 2, 2, 0]]).repeat(2, 1)
        target = torch.randint(3, 128, (2, 5))
        output = model(
            input_ids=source,
            attention_mask=source_mask,
            unit_ids=unit_ids,
            evidence_labels=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            decoder_input_ids=target,
            decoder_attention_mask=torch.ones_like(target),
            labels=target,
        )
        output["loss"].backward()
        injected = model.decoder.backbone.layers[-1]
        assert output["logits"].shape == (10, 128)
        assert torch.isfinite(output["loss"])
        assert injected.plan_gate.weight.grad is not None
        assert model.encoder.summary_tokens.grad is not None
