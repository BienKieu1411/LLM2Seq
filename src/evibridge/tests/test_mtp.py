import tempfile
import types
from pathlib import Path

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from evibridge.generation import autoregressive_generate
from evibridge.model import EviBridgeSeq2Seq
from evibridge.mtp import (
    CascadedFuturePredictor,
    future_prediction_loss,
    load_mtp_checkpoint,
    save_mtp_checkpoint,
)
from evibridge.mtp_generation import verified_mtp_generate


def test_phase3_loss_and_compact_checkpoint():
    torch.manual_seed(3)
    predictor = CascadedFuturePredictor(
        16,
        {
            "num_draft_tokens": 2,
            "ffn_size": 32,
            "dropout": 0.0,
            "draft_head_rank": 4,
            "max_positions_per_sequence": 4,
            "checkpoint_dtype": "bfloat16",
        },
    )
    embedding = torch.nn.Embedding(32, 16)
    lm_head = torch.nn.Linear(16, 32, bias=False)
    embedding.requires_grad_(False)
    lm_head.requires_grad_(False)
    states = torch.randn(2, 7, 16)
    labels = torch.randint(0, 32, (2, 7))
    labels[1, -1] = -100
    loss, metrics = future_prediction_loss(
        predictor,
        states,
        labels,
        embedding,
        lm_head,
        pad_token_id=0,
        config={
            "max_positions_per_sequence": 4,
            "distill_weight": 0.8,
            "supervised_weight": 0.2,
            "depth_weights": [1.0, 0.5],
        },
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(metrics["mtp_positions"]) > 0
    assert predictor.blocks[0].fuse.weight.grad is not None
    assert predictor.draft_head.down_proj.weight.grad is not None
    assert lm_head.weight.grad is None

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "phase3_mtp.pt"
        save_mtp_checkpoint(
            predictor,
            path,
            {"checkpoint_dtype": "bfloat16"},
            "final.pt",
            epoch=1,
            global_step=2,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert all("codebook" not in name for name in payload["mtp_state_dict"])
        floating = [value for value in payload["mtp_state_dict"].values() if value.is_floating_point()]
        assert floating and all(value.dtype == torch.bfloat16 for value in floating)
        restored = CascadedFuturePredictor(
            16,
            {"num_draft_tokens": 2, "ffn_size": 32, "dropout": 0.0, "draft_head_rank": 4},
        )
        loaded = load_mtp_checkpoint(restored, path)
        assert loaded["base_checkpoint"] == "final.pt"


def test_verified_mtp_matches_ar_for_qwen35_hybrid_cache():
    torch.manual_seed(7)
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
            "objectives": {},
            "mtp": {
                "num_draft_tokens": 3,
                "ffn_size": 96,
                "dropout": 0.0,
                "draft_head_rank": 16,
            },
        }
        model = EviBridgeSeq2Seq(config, text_config.vocab_size).eval()
        predictor = model.enable_mtp().eval()
        source = torch.randint(0, 128, (1, 12))
        source_mask = torch.ones_like(source)
        unit_ids = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0]])
        kwargs = {
            "max_new_tokens": 12,
            "min_new_tokens": 0,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 3,
            "eos_token_id": None,
            "pad_token_id": 0,
            "bos_token_id": 1,
        }
        ar_ids = autoregressive_generate(
            model,
            source,
            source_mask,
            unit_ids,
            do_sample=False,
            use_cache=True,
            **kwargs,
        )
        mtp = verified_mtp_generate(
            model,
            source,
            source_mask,
            unit_ids,
            fallback_probe_steps=0,
            **kwargs,
        )
        assert torch.equal(mtp.generated_ids, ar_ids)
        assert mtp.metrics["replay_calls"] > 0

        # Force one correct draft to cover the all-accepted/no-rollback path.
        short_kwargs = {**kwargs, "max_new_tokens": 4, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0}
        short_ar = autoregressive_generate(
            model,
            source,
            source_mask,
            unit_ids,
            do_sample=False,
            use_cache=True,
            **short_kwargs,
        )

        def known_good_draft(self, decoder_state, main_token, embed_tokens, lm_head, constrain, prefix_tokens, maximum=None):
            return [short_ar[:, 1:2].clone()]

        predictor.draft = types.MethodType(known_good_draft, predictor)
        accepted = verified_mtp_generate(
            model,
            source,
            source_mask,
            unit_ids,
            fallback_probe_steps=0,
            **short_kwargs,
        )
        assert torch.equal(accepted.generated_ids, short_ar)
        assert accepted.metrics["accepted_drafts"] >= 1
