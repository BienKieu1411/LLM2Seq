import tempfile

import torch
from genbridge.generation import autoregressive_generate
from genbridge.model import GenBridgeSeq2Seq
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, Qwen3Config, Qwen3ForCausalLM


def test_tiny_qwen35_forward_backward_and_cached_decode():
    text_config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=128,
        use_cache=False,
        tie_word_embeddings=True,
    )
    with tempfile.TemporaryDirectory() as checkpoint:
        Qwen3_5ForCausalLM(text_config).save_pretrained(checkpoint)
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

        generated = autoregressive_generate(
            model,
            source,
            source_mask,
            unit_ids=unit_ids,
            max_new_tokens=2,
            eos_token_id=None,
            pad_token_id=0,
            decoder_prefix_ids=[1, 17, 18],
            use_cache=True,
        )
        # The fixed task prefix conditions generation but is excluded from the
        # returned predictions and therefore from ROUGE.
        assert generated.shape == (2, 2)


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
