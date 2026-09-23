"""Offline bridge contracts and a tiny end-to-end CE/generation probe."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from relational_key.config import load_config, validate_architecture
from relational_key.data.source_alignment import SOURCE_ALIGNMENT_KEYS
from relational_key.evaluation.generate import generate_greedy
from relational_key.modeling.bridge import RelationalKeyBridge
from relational_key.modeling.model import RelationalKeyModel
from relational_key.modeling.outputs import EncoderState
from relational_key.runtime import build_loaders
from relational_key.training.checkpoint import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def _tiny_config() -> dict:
    config = load_config(ROOT / "configs/relational_key_pubmed.yaml")
    config["model"]["encoder_name"] = "__tiny__"
    config["model"]["decoder_name"] = "__tiny__"
    config["model"]["gradient_checkpointing"] = False
    config["data"]["train_file"] = str(ROOT / "tests/fixtures/train.jsonl")
    config["data"]["validation_file"] = str(ROOT / "tests/fixtures/validation.jsonl")
    config["data"]["test_file"] = str(ROOT / "tests/fixtures/test.jsonl")
    config["data"]["encoder_prefix"] = "Article: "
    config["data"]["decoder_prompt"] = "Summarize."
    config["data"]["decoder_prefix"] = "Abstract: "
    config["data"]["decoder_chat_template"] = False
    config["data"]["max_source_length"] = 64
    config["data"]["max_target_length"] = 16
    config["architecture"]["relation_rank"] = 8
    config["decoder"]["grounded_copy"]["key_dim"] = 8
    config["training"]["batch_size"] = 2
    config["training"]["num_workers"] = 0
    config["training"]["validation_num_workers"] = 0
    return config


def test_pubmed_settings_match_xov_except_bridge_and_output():
    source_root = ROOT.parent / "xov_bridge/configs"
    source_base = yaml.safe_load((source_root / "xov_base.yaml").read_text())
    source_task = yaml.safe_load((source_root / "xov_pubmed.yaml").read_text())
    new_base = yaml.safe_load((ROOT / "configs/relational_key_base.yaml").read_text())
    new_task = yaml.safe_load((ROOT / "configs/relational_key_pubmed.yaml").read_text())
    for section in ("model", "encoder", "decoder", "training", "data", "generation"):
        assert source_base[section] == new_base[section]
    for section in ("model", "decoder", "training", "data", "generation"):
        assert source_task.get(section) == new_task.get(section)
    config = load_config(ROOT / "configs/relational_key_pubmed.yaml")
    assert config["training"]["batch_size"] == 84
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["architecture"]["bridge_mode"] == "relational_key"
    with pytest.raises(ValueError, match="Unknown"):
        validate_architecture({**config["architecture"], "lexical_rank": 256})


def test_source_pairs_change_only_retrieval_keys_and_respect_masks():
    torch.manual_seed(23)
    arch = load_config(ROOT / "configs/relational_key_pubmed.yaml")["architecture"]
    bridge = RelationalKeyBridge(7, 11, arch)
    direct = RelationalKeyBridge(7, 11, {**arch, "bridge_mode": "direct_projection"})
    direct.direct_projection.load_state_dict(bridge.direct_projection.state_dict())
    final = torch.randn(1, 6, 7)
    state = EncoderState(
        final,
        (),
        torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool),
        torch.tensor([[0, 1, 1, 1, 0, 0]], dtype=torch.bool),
    )
    output = bridge(state)
    baseline = direct(state)
    torch.testing.assert_close(output.value_memory, baseline.memory, rtol=0, atol=0)
    torch.testing.assert_close(output.copy_memory, baseline.memory, rtol=0, atol=0)
    assert torch.count_nonzero(output.memory[:, [0, 4, 5]] - baseline.memory[:, [0, 4, 5]]) == 0
    assert torch.count_nonzero(output.memory[:, 1:4] - baseline.memory[:, 1:4]) > 0
    relative = (output.memory - output.copy_memory).square().mean(-1).sqrt() / output.copy_memory.square().mean(
        -1
    ).sqrt().clamp_min(1e-8)
    assert float(relative.detach().max()) <= 0.201

    changed = deepcopy(state)
    changed.final = final.clone()
    changed.final[:, 3] += 4.0
    changed_output = bridge(changed)
    torch.testing.assert_close(changed_output.copy_memory[:, 2], output.copy_memory[:, 2], rtol=0, atol=0)
    assert not torch.equal(changed_output.memory[:, 2], output.memory[:, 2])
    # A masked gap prevents the pair operator from using an adjacent source row.
    changed.content_mask = state.content_mask.clone()
    changed.content_mask[:, 3] = False
    gap_output = bridge(changed)
    changed.final[:, 3] -= 100.0
    torch.testing.assert_close(bridge(changed).memory[:, 2], gap_output.memory[:, 2], rtol=0, atol=0)


def test_bfloat16_bridge_without_autocast_keeps_memory_dtypes_consistent():
    arch = load_config(ROOT / "configs/relational_key_pubmed.yaml")["architecture"]
    bridge = RelationalKeyBridge(8, 12, arch).to(torch.bfloat16)
    state = EncoderState(
        torch.randn(1, 4, 8, dtype=torch.bfloat16),
        (),
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([[0, 1, 1, 0]], dtype=torch.bool),
    )
    output = bridge(state)
    assert output.memory.dtype == output.value_memory.dtype == output.copy_memory.dtype == torch.bfloat16
    assert torch.isfinite(output.memory).all()
    assert torch.equal(output.value_memory, output.copy_memory)


def test_ce_gradients_checkpoint_and_post_projection_key_effect(tmp_path):
    torch.manual_seed(33)
    config = _tiny_config()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    args = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    model = RelationalKeyModel(config)
    bridge = model.encode_source(
        batch["input_ids"],
        batch["attention_mask"],
        batch["source_content_mask"],
        batch["decoder_prompt_ids"],
        batch["decoder_prompt_mask"],
        torch.full((2,), 4.0),
        **{key: batch[key] for key in SOURCE_ALIGNMENT_KEYS},
    )
    assert torch.equal(bridge.value_memory, bridge.copy_memory)
    assert bridge.copy_state is not None
    cross = model.decoder.backbone.layers[0].cross
    base_key, base_value = cross._memory_kv(bridge.copy_memory, bridge.copy_memory)
    active_key, active_value = cross._memory_kv(bridge.memory, bridge.value_memory)
    assert not torch.equal(active_key, base_key)
    torch.testing.assert_close(active_value, base_value, rtol=0, atol=0)
    torch.manual_seed(17)
    query_state = torch.randn(bridge.memory.shape[0], 1, cross.hidden_size)
    query = cross.q_norm(
        cross.q_proj(query_state).view(bridge.memory.shape[0], 1, cross.num_heads, cross.head_dim)
    ).transpose(1, 2)
    repeat = cross.num_heads // cross.num_kv_heads
    valid = bridge.memory_mask[:, None, None, :]
    base_attention = (
        (query @ base_key.repeat_interleave(repeat, dim=1).transpose(-1, -2))
        .masked_fill(~valid, -torch.inf)
        .softmax(-1)
    )
    active_attention = (
        (query @ active_key.repeat_interleave(repeat, dim=1).transpose(-1, -2))
        .masked_fill(~valid, -torch.inf)
        .softmax(-1)
    )
    assert (active_attention - base_attention).abs().max() > 1e-8

    optimizer = torch.optim.SGD(model.bridge.parameters(), lr=0.1)
    before = {name: parameter.detach().clone() for name, parameter in model.bridge.named_parameters()}
    result = model(**args)
    assert result.loss is not None and torch.isfinite(result.loss)
    result.loss.backward()
    for name, parameter in model.bridge.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in model.bridge.named_parameters():
        assert not torch.equal(before[name], parameter), name

    checkpoint = tmp_path / "last.pt"
    save_checkpoint(checkpoint, model, optimizer, config, epoch=1, step=1)
    restored = RelationalKeyModel(config)
    load_checkpoint(checkpoint, restored, config=config, restore_rng=False)
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**args).logits, restored(**args).logits)
    different = deepcopy(config)
    different["architecture"]["relation_rank"] += 1
    with pytest.raises(ValueError, match="architecture_spec"):
        load_checkpoint(checkpoint, restored, config=different, restore_rng=False)


def test_cached_and_uncached_greedy_generation_agree():
    torch.manual_seed(8)
    config = _tiny_config()
    loader = build_loaders(config, split="test", batch_size_override=2)["test"]
    loader.collate_fn.include_targets = False
    batch = next(iter(loader))
    model = RelationalKeyModel(config).eval()
    tokenizer = loader.collate_fn.decoder_tokenizer
    with torch.no_grad():
        compact_text, compact_ids = generate_greedy(model, batch, tokenizer, max_new_tokens=4, compact_finished=True)
        full_text, full_ids = generate_greedy(model, batch, tokenizer, max_new_tokens=4, compact_finished=False)
    assert compact_text == full_text
    assert torch.equal(compact_ids, full_ids)
