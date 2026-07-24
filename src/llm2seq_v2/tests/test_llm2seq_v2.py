from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from llm2seq_v2.adapter import StableTokenLayerFusion, SummaryAdapterV2
from llm2seq_v2.checkpoint import load_last_checkpoint, save_last_checkpoint
from llm2seq_v2.config import load_config
from llm2seq_v2.decoder import QwenCopiedCrossAttention
from llm2seq_v2.metrics import rouge_scores

from llm2seq_v2.data import greedy_evidence_labels


def test_main_config_is_last_only_and_faithful():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_0_6b.yaml")
    assert config["model"]["encoder_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert config["model"]["decoder_name"] == "Qwen/Qwen3-0.6B"
    assert config["adapter"]["num_bidirectional_layers"] == 4
    assert config["decoder"]["cross_attention_every"] == 1
    assert config["decoder"]["initialize_cross_from_self"] is True
    assert config["checkpoint"] == {
        "save_best": False,
        "save_each_epoch": False,
        "save_last": True,
    }


def test_hiroute_config_enables_one_three_bank_adapter():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_0_6b_hiroute.yaml")
    assert config["adapter"]["hierarchical_sentence_context"] is True
    assert config["adapter"]["depth_routed_memory"] is True
    assert config["adapter"]["num_bidirectional_layers"] == 8
    assert config["decoder"]["memory_bank_count"] == 3
    assert config["checkpoint"]["save_last"] is True
    assert config["checkpoint"]["save_best"] is False


def test_config_rejects_best_checkpoint(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
model: {encoder_name: e, decoder_name: d, hidden_size: 8}
adapter: {layer_fusion: true, fuse_layers: [-1], num_heads: 1}
decoder: {cross_attention_every: 1, cross_gate_init: 0.1}
training: {interface_warmup_epochs: 0, full_finetune_epochs: 1, batch_size: 1, gradient_accumulation_steps: 1}
data: {train_file: a, validation_file: b, test_file: c, max_source_length: 8, max_target_length: 4}
checkpoint: {save_best: true}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="last.pt only"):
        load_config(path)


def test_layer_fusion_has_stable_last_layer_prior():
    fusion = StableTokenLayerFusion(8, [-1, -2, -3, -4], dropout=0.0)
    hidden = tuple(torch.randn(2, 5, 8) for _ in range(5))
    output = fusion(hidden, torch.ones(2, 5, dtype=torch.long))
    assert output.shape == (2, 5, 8)
    assert fusion.last_mean_weights is not None
    assert int(fusion.last_mean_weights.argmax()) == 0
    assert torch.allclose(fusion.last_mean_weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_adapter_is_bidirectional_and_preserves_full_memory():
    torch.manual_seed(3)
    config = {
        "layer_fusion": True,
        "fuse_layers": [-1, -2],
        "projection_ffn_size": 16,
        "projection_gate_init": 0.1,
        "num_bidirectional_layers": 2,
        "num_heads": 2,
        "ffn_size": 32,
        "dropout": 0.0,
        "bidirectional_gate_init": 0.5,
        "use_salience": True,
        "salience_hidden_size": 8,
        "salience_gate_init": 0.1,
    }
    adapter = SummaryAdapterV2(8, 8, config).eval()
    base = [torch.randn(1, 4, 8) for _ in range(3)]
    mask = torch.ones(1, 4, dtype=torch.long)
    units = torch.tensor([[1, 1, 2, 2]])
    labels = torch.tensor([[1.0, 0.0]])
    first = adapter(tuple(base), mask, units, labels)
    changed = copy.deepcopy(base)
    changed[-1][:, -1] += 5.0
    second = adapter(tuple(changed), mask, units, labels)
    assert first.memory.shape == (1, 4, 8)
    assert first.attention_bias.shape == (1, 4)
    assert torch.isfinite(first.loss_salience)
    # Full attention lets a future source token change an earlier memory state.
    assert not torch.allclose(first.memory[:, 0], second.memory[:, 0])


def test_adapter_supports_a_wider_pretrained_decoder():
    adapter = SummaryAdapterV2(
        8,
        12,
        {
            "hidden_size": 8,
            "layer_fusion": False,
            "projection_ffn_size": 16,
            "num_bidirectional_layers": 1,
            "num_heads": 2,
            "ffn_size": 24,
            "dropout": 0.0,
            "use_salience": False,
        },
    )
    output = adapter(
        (torch.randn(2, 5, 8),),
        torch.ones(2, 5, dtype=torch.long),
    )
    assert output.memory.shape == (2, 5, 12)
    assert isinstance(adapter.memory_projection, nn.Sequential)


def test_hiroute_adapter_preserves_three_full_length_memories():
    torch.manual_seed(7)
    adapter = SummaryAdapterV2(
        8,
        8,
        {
            "hidden_size": 8,
            "layer_fusion": True,
            "fuse_layers": [-1, -2],
            "projection_ffn_size": 16,
            "num_bidirectional_layers": 1,
            "num_heads": 2,
            "ffn_size": 24,
            "dropout": 0.0,
            "use_salience": True,
            "salience_hidden_size": 8,
            "depth_routed_memory": True,
            "lexical_layers": [-3, -2],
            "semantic_layers": [-2, -1],
            "branch_projection_ffn_size": 16,
            "hierarchical_sentence_context": True,
            "sentence_context_size": 8,
            "sentence_context_heads": 2,
            "sentence_context_ffn_size": 16,
            "sentence_context_layers": 1,
            "sentence_broadcast_gate_init": 0.1,
        },
    ).eval()
    output = adapter(
        tuple(torch.randn(2, 6, 8) for _ in range(4)),
        torch.ones(2, 6, dtype=torch.long),
        torch.tensor([[1, 1, 1, 2, 2, 2], [1, 1, 2, 2, 3, 3]]),
        torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    )
    assert output.memory.shape == (2, 3, 6, 8)
    assert output.attention_bias.shape == (2, 6)
    assert torch.isfinite(output.memory).all()
    assert torch.isfinite(output.loss_salience)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.q_norm = nn.RMSNorm(4)
        self.k_norm = nn.RMSNorm(4)


def test_cross_attention_copies_all_native_projections():
    source = _Attention()
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        initializer_range=0.02,
    )
    cross = QwenCopiedCrossAttention(source, nn.RMSNorm(8), config, 0.0, True)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert torch.equal(getattr(cross, name).weight, getattr(source, name).weight)
        assert getattr(cross, name).weight.data_ptr() != getattr(source, name).weight.data_ptr()
    output = cross(
        torch.randn(2, 3, 8),
        torch.randn(2, 5, 8),
        torch.ones(2, 5, dtype=torch.long),
        None,
    )
    assert output.shape == (2, 3, 8)


def test_checkpoint_is_complete_last_only(tmp_path: Path):
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    path = tmp_path / "last.pt"
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    save_last_checkpoint(model, path, {"x": 1}, 15, 123, {"train": {"sha256": "x"}})
    for parameter in model.parameters():
        parameter.data.zero_()
    payload = load_last_checkpoint(model, path)
    assert payload["checkpoint_role"] == "last"
    assert payload["stores_full_model_state"]
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    with pytest.raises(ValueError):
        save_last_checkpoint(model, tmp_path / "best.pt", {}, 1, 1, {})


def test_oracle_and_locked_rouge():
    labels = greedy_evidence_labels(
        ["Mở cài đặt.", "Khởi động lại thiết bị.", "Mua một chiếc bàn."],
        "Khởi động lại thiết bị.",
    )
    assert labels == [0.0, 1.0, 0.0]
    scores = rouge_scores(["khởi động lại thiết bị"], ["Khởi động lại thiết bị"])
    assert scores["rouge1"] > 99.9
    assert scores["rouge2"] > 99.9
