"""Behavioral checks for active initialization and repaired source alignment."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from xov.config import load_config
from xov.data.source_alignment import align_source_tokens, pad_source_alignments, SOURCE_ALIGNMENT_KEYS
from xov.modeling.model import XOVModel
from xov.modeling.outputs import EncoderState
from xov.modeling.xov import CrossTokenizerOrderedValueBridge
from xov.runtime import build_loaders
from xov.training.checkpoint import architecture_spec, save_checkpoint, load_checkpoint

CONFIG = Path(__file__).resolve().parents[1] / "configs/xov_smoke.yaml"


def test_gap_mask_matches_contiguous_conv_and_blocks_false_neighbors():
    bridge = CrossTokenizerOrderedValueBridge(8, 8, load_config(CONFIG)["architecture"])
    x = torch.randn(2, 4, 8)
    valid = torch.ones(2, 4, dtype=torch.bool)
    positions = torch.arange(4).expand(2, -1)
    torch.testing.assert_close(
        bridge._ordered_features(x, positions, valid), bridge.phrase_conv(x.transpose(1, 2)).transpose(1, 2)
    )
    positions = torch.tensor([[0, 1, 5, 6], [0, 1, 5, 6]])
    altered = x.clone()
    altered[:, 2:] += 100
    torch.testing.assert_close(
        bridge._ordered_features(x, positions, valid)[:, :2], bridge._ordered_features(altered, positions, valid)[:, :2]
    )


def test_original_positions_and_prefix_crossing():
    class Tokenizer:
        all_special_ids = (2,)

        def __call__(self, text, **kwargs):
            return {"input_ids": [10, 2, 11], "offset_mapping": [(0, 1), (1, 2), (2, 3)]}

    aligned = align_source_tokens("a#b", 2, [(1, 3), (3, 4), (4, 5)], Tokenizer())
    assert aligned["copy_token_positions"] == [0, 2]
    assert aligned["copy_encoder_indices"] == [0, 2]
    padded = pad_source_alignments([aligned, {key: [] for key in aligned}])
    assert padded["copy_token_positions"].shape == padded["copy_token_ids"].shape
    assert not padded["copy_token_mask"][1].any()


@pytest.mark.parametrize("copy_enabled", [True, False])
@pytest.mark.parametrize("gate_mode", ["global", "source_lexical"])
def test_real_ce_first_step_updates_branch_and_checkpoint_roundtrip(tmp_path, copy_enabled, gate_mode):
    torch.manual_seed(33)
    config = load_config(CONFIG)
    config["decoder"]["grounded_copy"]["enabled"] = copy_enabled
    config["architecture"]["value_gate_mode"] = gate_mode
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    model = XOVModel(config)
    args = {k: v for k, v in batch.items() if torch.is_tensor(v)}
    before = {k: p.detach().clone() for k, p in model.bridge.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = model(**args)
    assert not torch.equal(result.bridge.value_memory, result.bridge.memory)
    result.loss.backward()
    for name, parameter in model.bridge.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in model.bridge.named_parameters():
        assert not torch.equal(before[name], parameter), name
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, config, epoch=1, step=1)
    restored = XOVModel(config)
    load_checkpoint(path, restored, config=config, restore_rng=False)
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**args).logits, restored(**args).logits)
    state = torch.load(path, weights_only=False)
    state["architecture_spec"]["operator_contract"] = "old_conv_pool_silu"
    torch.save(state, path)
    with pytest.raises(ValueError, match="architecture_spec"):
        load_checkpoint(path, restored, config=config, restore_rng=False)
    for key in SOURCE_ALIGNMENT_KEYS:
        args.pop(key)
    with pytest.raises(ValueError, match="alignment is missing"):
        model(**args)


def test_source_lexical_gate_starts_at_global_then_can_vary_by_source():
    torch.manual_seed(19)
    global_config = load_config(CONFIG)["architecture"]
    local_config = dict(global_config, value_gate_mode="source_lexical")
    global_bridge = CrossTokenizerOrderedValueBridge(8, 8, global_config)
    local_bridge = CrossTokenizerOrderedValueBridge(8, 8, local_config)
    local_bridge.load_state_dict(global_bridge.state_dict(), strict=False)
    embedding = torch.nn.Embedding(32, 8)
    state = EncoderState(
        torch.randn(1, 3, 8),
        (),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
    )
    alignment = pad_source_alignments(
        [
            dict(
                copy_token_ids=[4, 5, 6],
                copy_token_positions=[0, 1, 2],
                copy_encoder_indices=[0, 1, 2],
                copy_token_indices=[0, 1, 2],
                copy_alignment_weights=[1.0, 1.0, 1.0],
            )
        ]
    )
    global_values = global_bridge(state, embedding, **alignment).value_memory
    local_values = local_bridge(state, embedding, **alignment).value_memory
    torch.testing.assert_close(local_values, global_values, rtol=0, atol=0)
    with torch.no_grad():
        local_bridge.source_gate.weight[0, 0] = 3.0
    changed = local_bridge(state, embedding, **alignment)
    assert not torch.equal(changed.value_memory, global_values)
    assert torch.equal(changed.memory, global_bridge(state, embedding, **alignment).memory)
    assert torch.equal(changed.copy_memory, state.final)


def test_source_lexical_gate_reaches_decoder_without_changing_copy_keys():
    torch.manual_seed(29)
    config = load_config(CONFIG)
    config["architecture"]["value_gate_mode"] = "source_lexical"
    config["decoder"]["grounded_copy"]["enabled"] = True
    model = XOVModel(config).eval()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    args = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    with torch.no_grad():
        before = model(**args)
        model.bridge.source_gate.weight[0, 0] = 10.0
        after = model(**args)
    assert torch.equal(before.bridge.memory, after.bridge.memory)
    assert torch.equal(before.bridge.copy_state.keys, after.bridge.copy_state.keys)
    assert not torch.equal(before.bridge.value_memory, after.bridge.value_memory)
    assert not torch.equal(before.logits, after.logits)


def test_key_gate_is_independent_and_leaves_copy_route_anchored():
    torch.manual_seed(37)
    config = load_config(CONFIG)
    config["decoder"]["grounded_copy"]["enabled"] = True
    model = XOVModel(config).eval()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    args = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    with torch.no_grad():
        before = model(**args)
        model.bridge.key_gate_raw.fill_(5.0)
        after = model(**args)
    assert not torch.equal(before.bridge.memory, after.bridge.memory)
    assert torch.equal(before.bridge.value_memory, after.bridge.value_memory)
    assert torch.equal(before.bridge.copy_memory, after.bridge.copy_memory)
    assert torch.equal(before.bridge.copy_state.keys, after.bridge.copy_state.keys)
    assert not torch.equal(before.logits, after.logits)


@pytest.mark.parametrize("encoder_hidden", [8, 6])
def test_source_lexical_gate_runs_in_bf16_without_autocast(encoder_hidden):
    torch.manual_seed(43)
    config = dict(load_config(CONFIG)["architecture"], value_gate_mode="source_lexical")
    bridge = CrossTokenizerOrderedValueBridge(encoder_hidden, 8, config).to(torch.bfloat16)
    embedding = torch.nn.Embedding(32, 8).to(torch.bfloat16)
    state = EncoderState(
        torch.randn(1, 3, encoder_hidden, dtype=torch.bfloat16),
        (),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
    )
    alignment = pad_source_alignments(
        [
            dict(
                copy_token_ids=[4, 5, 6],
                copy_token_positions=[0, 1, 2],
                copy_encoder_indices=[0, 1, 2],
                copy_token_indices=[0, 1, 2],
                copy_alignment_weights=[1.0, 1.0, 1.0],
            )
        ]
    )
    result = bridge(state, embedding, **alignment)
    assert result.memory.isfinite().all()
    assert result.value_memory.isfinite().all()
    projection_dtype = (
        bridge.direct_projection.weight.dtype
        if isinstance(bridge.direct_projection, torch.nn.Linear)
        else torch.float32
    )
    expected_anchor = bridge.direct_projection(state.final.to(projection_dtype))
    torch.testing.assert_close(result.copy_memory, expected_anchor)


@pytest.mark.parametrize("autocast", [False, True])
def test_cap_padding_and_many_to_one_order(autocast):
    torch.manual_seed(7)
    config = load_config(CONFIG)["architecture"]
    bridge = CrossTokenizerOrderedValueBridge(24, 24, config)
    embedding = torch.nn.Embedding(128, 24)
    state = EncoderState(torch.randn(1, 3, 24), (), torch.tensor([[1, 1, 0]]).bool(), torch.tensor([[0, 1, 0]]).bool())
    align = pad_source_alignments(
        [
            dict(
                copy_token_ids=[4, 5, 6, 7],
                copy_token_positions=[0, 1, 2, 3],
                copy_encoder_indices=[1, 1, 1, 1],
                copy_token_indices=[0, 1, 2, 3],
                copy_alignment_weights=[1.0, 1.0, 1.0, 1.0],
            )
        ]
    )
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        result = bridge(state, embedding, **align)
        swapped = deepcopy(align)
        swapped["copy_token_ids"] = swapped["copy_token_ids"][:, [0, 2, 1, 3]]
        other = bridge(state, embedding, **swapped)
    value_delta = result.value_memory - result.copy_memory
    key_delta = result.memory - result.copy_memory
    assert torch.count_nonzero(value_delta[:, [0, 2]]) == 0
    assert torch.count_nonzero(key_delta[:, [0, 2]]) == 0
    assert value_delta[:, 1].norm() > 0
    assert key_delta[:, 1].norm() > 0
    assert not torch.equal(result.value_memory, other.value_memory)
    assert not torch.equal(result.memory, other.memory)
    assert value_delta.norm(dim=-1).max() <= 0.201 * result.copy_memory.norm(dim=-1).max()
    assert key_delta.norm(dim=-1).max() <= 0.201 * result.copy_memory.norm(dim=-1).max()
    (result.value_memory.square().mean() + result.memory.square().mean()).backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in bridge.parameters())


def test_key_route_changes_post_projection_attention_weights_with_fixed_query():
    torch.manual_seed(41)
    config = load_config(CONFIG)
    model = XOVModel(config).eval()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    with torch.no_grad():
        bridge = model.encode_source(
            batch["input_ids"],
            batch["attention_mask"],
            batch["source_content_mask"],
            batch["decoder_prompt_ids"],
            batch["decoder_prompt_mask"],
            torch.full((batch["input_ids"].shape[0],), 4.0),
            **{key: batch[key] for key in SOURCE_ALIGNMENT_KEYS},
        )
        cross = model.decoder.backbone.layers[0].cross
        query_states = torch.randn(bridge.memory.shape[0], 1, cross.hidden_size)
        query = cross.q_norm(
            cross.q_proj(query_states).view(bridge.memory.shape[0], 1, cross.num_heads, cross.head_dim)
        ).transpose(1, 2)
        base_key, _ = cross._memory_kv(bridge.copy_memory, bridge.value_memory)
        active_key, _ = cross._memory_kv(bridge.memory, bridge.value_memory)
        repeat = cross.num_heads // cross.num_kv_heads
        base_logits = query @ base_key.repeat_interleave(repeat, dim=1).transpose(-1, -2)
        active_logits = query @ active_key.repeat_interleave(repeat, dim=1).transpose(-1, -2)
        mask = bridge.memory_mask[:, None, None, :]
        base_attention = base_logits.masked_fill(~mask, -torch.inf).softmax(dim=-1)
        active_attention = active_logits.masked_fill(~mask, -torch.inf).softmax(dim=-1)
    assert not torch.equal(active_key, base_key)
    assert (active_attention - base_attention).abs().max() > 1e-7
    torch.testing.assert_close(
        bridge.copy_memory,
        model.bridge.direct_projection(
            model.encoder(batch["input_ids"], batch["attention_mask"], batch["source_content_mask"]).final
        ).masked_fill(~bridge.memory_mask[..., None], 0.0),
    )


def test_cached_teacher_forcing_guard():
    model = XOVModel(load_config(CONFIG))
    with pytest.raises(ValueError, match="use_cache=False"):
        model.decoder(
            input_ids=torch.ones(1, 2, dtype=torch.long),
            memory=torch.zeros(1, 3, 24),
            memory_mask=torch.ones(1, 3, dtype=torch.bool),
            labels=torch.ones(1, 2, dtype=torch.long),
            use_cache=True,
        )


def _empty_rank_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(39)
        config = load_config(CONFIG)
        config["architecture"]["value_gate_mode"] = "source_lexical"
        model = XOVModel(config)
        ddp = DistributedDataParallel(model, find_unused_parameters=False)
        batch = next(iter(build_loaders(config, max_train_examples=4)["train"]))
        args = {key: value for key, value in batch.items() if torch.is_tensor(value)}
        args["return_logits"] = False
        if rank == 1:
            args["copy_token_mask"].zero_()
            args["copy_alignment_weights"].zero_()
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
        original = model.bridge.lexical_up.weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            ddp(**args).loss.backward()
            assert all(p.grad is not None and p.grad.isfinite().all() for p in model.bridge.parameters())
            optimizer.step()
        assert not torch.equal(original, model.bridge.lexical_up.weight)
        gathered = [torch.zeros_like(original) for _ in range(2)]
        dist.all_gather(gathered, model.bridge.lexical_up.weight.detach())
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_ddp_empty_alignment_rank_keeps_updates_synchronized(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_empty_rank_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("data", "encoder_prefix", "Changed prefix: "),
        ("data", "max_source_length", 3072),
        ("data", "decoder_prefix", "New output: "),
        ("data", "detokenize", True),
        ("model", "tokenizer_use_fast", False),
        ("model", "attention_implementation", "eager"),
        ("decoder", "attention_dropout", 0.1),
    ],
)
def test_checkpoint_rejects_changed_input_or_execution_policy(tmp_path, section, key, value):
    config = load_config(CONFIG)
    model = XOVModel(config)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, None, config, epoch=0, step=0)
    changed = deepcopy(config)
    changed[section][key] = value
    with pytest.raises(ValueError, match="architecture_spec"):
        load_checkpoint(path, model, config=changed, restore_rng=False)


def test_direct_checkpoint_spec_keeps_historical_graph_contract():
    config = load_config(CONFIG)
    config["architecture"]["bridge_mode"] = "direct_projection"
    spec = architecture_spec(config)
    assert spec["graph_version"] == "ordered_lexical_values"
    assert spec["operator_contract"] == "silu_before_pool_original_adjacency_clipped_source_copy_anchor"
    assert spec["key_memory"] == "direct_projection"
    assert "key_gate_init" not in spec
    assert "key_gate_max" not in spec

    config["architecture"]["bridge_mode"] = "cross_tokenizer_ordered_value"
    active = architecture_spec(config)
    assert active["graph_version"] == "ordered_lexical_keys_and_values"
    assert active["key_memory"] == "direct_projection_anchor_plus_bounded_lexical"
    assert active["key_gate_init"] == 0.12
    assert active["key_gate_max"] == 0.20
