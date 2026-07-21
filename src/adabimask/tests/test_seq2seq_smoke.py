import tempfile

import torch
from adabimask.backbone import load_text_causal_lm
from adabimask.direct_baseline import DirectCausalBaseline
from adabimask.encoder import AdaBiMaskEncoder
from adabimask.model import AdaBiMaskSeq2Seq
from adabimask.pretrained_decoder import CrossAttentionInjectedLayer
from transformers import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)


def test_complete_seq2seq_builds_from_a_qwen35_checkpoint():
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
                "use_lora": False,
                "train_base": True,
                "gradient_checkpointing": False,
            },
            "mask": {
                "mode": "learnable",
                "num_groups": 1,
                "budget_groups": 1,
                "init_probability": 0.01,
            },
            "decoder": {
                "pretrained_name": checkpoint,
                "num_layers": 4,
                "gradient_checkpointing": False,
            },
        }
        model = AdaBiMaskSeq2Seq(config, vocab_size=text_config.vocab_size)
        model.set_training_stage("full_finetune")
        source_ids = torch.randint(0, text_config.vocab_size, (2, 10))
        target_ids = torch.randint(0, text_config.vocab_size, (2, 6))
        labels = target_ids.clone()
        labels[0, -2:] = -100
        outputs = model(
            input_ids=source_ids,
            attention_mask=torch.ones_like(source_ids),
            decoder_input_ids=target_ids,
            decoder_attention_mask=torch.ones_like(target_ids),
            labels=labels,
        )
        outputs["loss"].backward()
        assert outputs["logits"].shape == (10, text_config.vocab_size)
        assert torch.isfinite(outputs["loss"])
        assert model.encoder.policy.gate_logits.grad is not None

        with torch.no_grad():
            inference_outputs = model(
                input_ids=source_ids,
                attention_mask=torch.ones_like(source_ids),
                decoder_input_ids=target_ids,
                decoder_attention_mask=torch.ones_like(target_ids),
            )
        assert inference_outputs["logits"].shape == (2, 6, text_config.vocab_size)

        model.eval()
        memory = model.encode(source_ids, torch.ones_like(source_ids))
        first_states, cache = model.decoder(
            input_ids=target_ids[:, :1],
            encoder_hidden_states=memory,
            encoder_attention_mask=torch.ones_like(source_ids),
            use_cache=True,
        )
        next_states, cache = model.decoder(
            input_ids=target_ids[:, 1:2],
            encoder_hidden_states=memory,
            encoder_attention_mask=torch.ones_like(source_ids),
            past_key_values=cache,
            use_cache=True,
        )
        assert first_states.shape == next_states.shape == (2, 1, text_config.hidden_size)
        assert cache is not None


def test_multimodal_qwen35_checkpoint_loads_as_text_only():
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
        tie_word_embeddings=False,
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=64,
        out_hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_position_embeddings=64,
    )
    outer_config = Qwen3_5Config(text_config=text_config, vision_config=vision_config)
    with tempfile.TemporaryDirectory() as checkpoint:
        original = Qwen3_5ForConditionalGeneration(outer_config)
        expected_embedding = original.model.language_model.embed_tokens.weight.detach().clone()
        original.save_pretrained(checkpoint)
        loaded, loaded_config = load_text_causal_lm(checkpoint, dtype=torch.float32)
        assert loaded_config.model_type == "qwen3_5_text"
        assert not hasattr(loaded.model, "visual")
        assert torch.equal(loaded.model.embed_tokens.weight, expected_embedding)


def test_new_cross_attention_uses_the_pretrained_backbone_dtype():
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
    )
    backbone = Qwen3_5ForCausalLM(text_config).to(torch.bfloat16).model
    source_attention = backbone.layers[-1].self_attn
    wrapped = CrossAttentionInjectedLayer(
        backbone.layers[0],
        text_config,
        source_attention,
        dropout=0.0,
    )
    assert wrapped.cross_attn.q_proj.weight.dtype == torch.bfloat16
    assert wrapped.cross_attn.k_proj.weight.dtype == torch.bfloat16
    assert wrapped.cross_attn.o_proj.weight.dtype == torch.bfloat16


def test_qwen35_hybrid_encoder_accepts_the_lora_ablation_targets():
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
    )
    with tempfile.TemporaryDirectory() as checkpoint:
        Qwen3_5ForCausalLM(text_config).save_pretrained(checkpoint)
        encoder = AdaBiMaskEncoder(
            {
                "encoder_name": checkpoint,
                "dtype": "float32",
                "use_lora": True,
                "train_base": False,
                "gradient_checkpointing": False,
                "lora": {
                    "r": 4,
                    "alpha": 8,
                    "target_modules": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "in_proj_qkv",
                        "in_proj_z",
                        "in_proj_b",
                        "in_proj_a",
                        "out_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                },
            },
            {"mode": "causal", "num_groups": 1, "budget_groups": 1},
        )
        trainable = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
        assert trainable
        assert all("lora_" in name for name in trainable)


def test_direct_baseline_projects_only_supervised_summary_positions():
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
    )
    with tempfile.TemporaryDirectory() as checkpoint:
        Qwen3_5ForCausalLM(text_config).save_pretrained(checkpoint)
        model = DirectCausalBaseline(
            {
                "encoder_name": checkpoint,
                "dtype": "float32",
                "use_lora": False,
                "train_base": True,
                "gradient_checkpointing": False,
            }
        )
        input_ids = torch.randint(0, text_config.vocab_size, (2, 12))
        labels = input_ids.clone()
        labels[:, :8] = -100
        output = model(input_ids, torch.ones_like(input_ids), labels=labels)
        assert output["logits"].shape == (8, text_config.vocab_size)
        output["loss"].backward()
        assert torch.isfinite(output["loss"])
