import tempfile

import torch
from evibridge.model import EviBridgeSeq2Seq
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


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
                "use_lora": False,
                "train_base": True,
                "gradient_checkpointing": False,
                "layer_fusion": {"enabled": False},
            },
            "bridge": {
                "mode": "evidence",
                "hidden_size": 32,
                "num_layers": 1,
                "num_heads": 4,
                "ffn_size": 64,
                "num_evidence_slots": 2,
                "dropout": 0.0,
            },
            "decoder": {
                "pretrained_name": checkpoint,
                "num_layers": 4,
                "gradient_checkpointing": False,
            },
            "objectives": {"evidence_weight": 0.2, "diversity_weight": 0.01},
        }
        model = EviBridgeSeq2Seq(config, text_config.vocab_size)
        model.set_training_stage("full_finetune")
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
        assert model.bridge.slot_queries.grad is not None

        model.eval()
        memory, memory_mask = model.encode(source, source_mask, unit_ids, return_attention_mask=True)
        first, cache = model.decoder(
            input_ids=target[:, :1],
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            use_cache=True,
        )
        second, cache = model.decoder(
            input_ids=target[:, 1:2],
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            past_key_values=cache,
            use_cache=True,
        )
        assert first.shape == second.shape == (2, 1, 64)
        assert cache is not None
