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
from xov.training.checkpoint import save_checkpoint, load_checkpoint

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
def test_real_ce_first_step_updates_branch_and_checkpoint_roundtrip(tmp_path, copy_enabled):
    torch.manual_seed(33)
    config = load_config(CONFIG)
    config["decoder"]["grounded_copy"]["enabled"] = copy_enabled
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
    delta = result.value_memory - result.memory
    assert torch.count_nonzero(delta[:, [0, 2]]) == 0
    assert delta[:, 1].norm() > 0
    assert not torch.equal(result.value_memory, other.value_memory)
    assert delta.norm(dim=-1).max() <= 0.201 * result.memory.norm(dim=-1).max()
    result.value_memory.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in bridge.parameters())


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
