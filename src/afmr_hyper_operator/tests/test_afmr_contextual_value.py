import copy
import re
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from eviseq_afmr.config import contextual_value_settings, hyper_operator_settings, load_config, validate_config
from eviseq_afmr.data.copy_alignment import COPY_INPUT_KEYS
from eviseq_afmr.evaluation.generate import generate_greedy, generate_sampled
from eviseq_afmr.modeling.contextual_value import ContextualValueBridge
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import build_loaders
from eviseq_afmr.training.checkpoint import architecture_spec, load_checkpoint, save_checkpoint
from eviseq_afmr.training.optimizer import build_optimizer, set_stage_trainability


def _config():
    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    config["architecture"]["contextual_value"].update(enabled=True, dim=16, num_heads=4, window_size=4, stride=2)
    config["decoder"]["grounded_copy"]["enabled"] = True
    config["decoder"]["ce_chunk_size"] = 8
    return config


def _setup(config=None):
    config = _config() if config is None else config
    loader = build_loaders(config, max_train_examples=2)["train"]
    return config, EviSeqAFMR(config), next(iter(loader)), loader.collate_fn.decoder_tokenizer


def _forward(model, batch, logits=False):
    return model(**{k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}, return_logits=logits)


def _bridge(model, batch):
    return model.encode_source(
        *(
            batch[k]
            for k in ("input_ids", "attention_mask", "source_content_mask", "decoder_prompt_ids", "decoder_prompt_mask")
        ),
        torch.full((batch["input_ids"].shape[0],), 32.0),
        **{key: batch[key] for key in COPY_INPUT_KEYS},
    )


def test_zero_initialization_preserves_baseline_weights_rng_and_outputs():
    config = _config()
    old_config = copy.deepcopy(config)
    old_config["architecture"].pop("contextual_value")
    torch.manual_seed(13)
    old = EviSeqAFMR(old_config).eval()
    old_rng = torch.get_rng_state()
    torch.manual_seed(13)
    new = EviSeqAFMR(config).eval()
    torch.testing.assert_close(torch.get_rng_state(), old_rng, rtol=0, atol=0)
    for name, tensor in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], tensor, rtol=0, atol=0)
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    with torch.no_grad():
        before, after = _forward(old, batch, True), _forward(new, batch, True)
    torch.testing.assert_close(after.logits, before.logits, rtol=0, atol=0)
    torch.testing.assert_close(after.loss, before.loss, rtol=0, atol=0)
    assert after.bridge.value_residual.count_nonzero() == 0


def test_region_context_is_content_only_padding_invariant_and_global():
    torch.manual_seed(3)
    settings = contextual_value_settings(_config()["architecture"])
    module = ContextualValueBridge(24, settings).eval()
    torch.nn.init.normal_(module.output.weight, std=0.1)
    anchor = torch.randn(3, 14, 24, requires_grad=True)
    mask = torch.zeros(3, 14, dtype=torch.bool)
    mask[0, 2:12] = True
    mask[2, 3] = True
    first = module(anchor, mask)
    altered = anchor.detach().clone().masked_fill(~mask[..., None], 10000)
    torch.testing.assert_close(first, module(altered, mask), atol=1e-6, rtol=1e-5)
    compact = anchor[0, mask[0]][None]
    torch.testing.assert_close(first[0, mask[0]], module(compact, torch.ones(1, 10, dtype=torch.bool))[0])
    assert torch.isfinite(first).all() and first[~mask].count_nonzero() == 0
    altered[0, 10:12] += 10 * torch.randn(2, 24)
    assert not torch.allclose(first[0, 2], module(altered, mask)[0, 2])
    first.square().sum().backward()
    assert torch.isfinite(anchor.grad).all() and anchor.grad[~mask].count_nonzero() == 0


def test_region_mapping_is_local_and_overlap_is_averaged():
    content = torch.tensor([[False, True, True, False, True, True, True, True, False]])
    regions = torch.tensor([[[2.0, 2.0], [6.0, 6.0]]], requires_grad=True)
    mapped = ContextualValueBridge._map_regions_to_tokens(
        regions, torch.tensor([[True, True]]), torch.tensor([[0, 2]]), torch.tensor([[4, 6]]), content
    )
    expected = torch.tensor([0.0, 2.0, 2.0, 0.0, 4.0, 4.0, 6.0, 6.0, 0.0])
    torch.testing.assert_close(mapped[0, :, 0], expected)
    torch.testing.assert_close(mapped[0, :, 1], expected)
    mapped[0, 1].sum().backward()
    assert regions.grad[0, 0].abs().sum() > 0
    assert regions.grad[0, 1].count_nonzero() == 0


def test_context_residual_has_no_common_content_offset():
    torch.manual_seed(21)
    settings = contextual_value_settings(_config()["architecture"])
    module = ContextualValueBridge(24, settings).eval()
    torch.nn.init.normal_(module.output.weight, std=0.1)
    anchor = torch.randn(2, 9, 24)
    content = torch.tensor([[False, True, True, True, False, True, True, False, False], [False] * 9])
    residual = module(anchor, content)
    torch.testing.assert_close(residual[0, content[0]].mean(0), torch.zeros(24), atol=1e-6, rtol=0)
    assert residual[0, ~content[0]].count_nonzero() == 0
    assert residual[1].count_nonzero() == 0
    assert torch.isfinite(residual).all()


@pytest.mark.parametrize("bf16", [False, True])
def test_projected_value_bound_and_gradients_at_zero_large_and_zero_anchor(bf16):
    _, model, _, _ = _setup()
    cross = model.decoder.backbone.layers[0].cross
    memory, anchor = torch.randn(2, 9, 24), torch.randn(2, 9, 24)
    for size in (0.0, 10000.0):
        delta = (torch.randn_like(anchor) * size).requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            key, base = cross._memory_kv(memory, anchor)
            new_key, value = cross._memory_kv(memory, anchor, delta)
        torch.testing.assert_close(key, new_key, rtol=0, atol=0)
        rms_delta = (value.float() - base.float()).square().mean(-1).sqrt()
        bound = base.float().square().mean(-1).sqrt() * 0.20
        assert torch.all(rms_delta <= bound * (1.04 if bf16 else 1.00001) + 1e-7)
        value.float().square().sum().backward()
        assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0
        if size == 0:
            torch.testing.assert_close(value, base, rtol=0, atol=0)
    delta = torch.randn_like(anchor, requires_grad=True)
    _, value = cross._memory_kv(memory, torch.zeros_like(anchor), delta)
    assert value.count_nonzero() == 0
    value.sum().backward()
    assert torch.isfinite(delta.grad).all()


def test_value_context_changes_decoder_but_preserves_copy_anchor_and_keys():
    torch.manual_seed(9)
    _, model, batch, _ = _setup()
    model.eval()
    with torch.no_grad():
        before = _forward(model, batch, True)
        torch.nn.init.normal_(model.bridge.contextual_value.output.weight, std=0.2)
        after = _forward(model, batch, True)
        for name in ("memory", "value_memory", "source_bias"):
            torch.testing.assert_close(getattr(before.bridge, name), getattr(after.bridge, name), rtol=0, atol=0)
        for name in ("keys", "token_ids", "mask", "bias"):
            torch.testing.assert_close(
                getattr(before.bridge.copy_state, name), getattr(after.bridge.copy_state, name), rtol=0, atol=0
            )
        assert not torch.allclose(after.logits, before.logits)
        torch.testing.assert_close(after.loss, _forward(model, batch).loss)


@pytest.mark.parametrize("stage", ["interface_warmup", "full_finetune"])
@pytest.mark.parametrize("bf16", [False, True])
def test_checkpointed_ce_updates_all_context_parameters_and_copy(stage, bf16):
    torch.manual_seed(5)
    config, model, batch, _ = _setup()
    config["training"]["weight_decay"] = 0.0
    set_stage_trainability(model, stage)
    optimizer = build_optimizer(model, config, stage)
    branch = model.bridge.contextual_value
    parameters = dict(branch.named_parameters())
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        before = {name: p.detach().clone() for name, p in parameters.items()}
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            output = _forward(model, batch)
        output.loss.backward()
        checked = parameters if step else {"output.weight": branch.output.weight}
        for name, parameter in checked.items():
            assert parameter.dtype == parameter.grad.dtype == torch.float32, name
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name
        for parameter in (
            model.decoder.grounded_copy.gate.weight,
            model.decoder.backbone.layers[0].cross.v_proj.weight,
        ):
            assert (
                parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
            )
        if stage == "full_finetune":
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
        else:
            assert all(p.grad is None for p in model.encoder.parameters())
        optimizer.step()
        for name, parameter in checked.items():
            assert not torch.equal(parameter, before[name]), name


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_value_context_cache_compaction_sampling_and_checkpoint_roundtrip(dtype, tmp_path):
    torch.manual_seed(15)
    config = _config()
    config["model"]["dtype"] = dtype
    config, model, batch, tokenizer = _setup(config)
    model.eval()
    torch.nn.init.normal_(model.bridge.contextual_value.output.weight, std=0.2)
    decoder = model.decoder
    with torch.no_grad():
        state = _bridge(model, batch)
        assert state.value_residual.dtype == state.value_memory.dtype == decoder.embed_tokens.weight.dtype
        tokens = torch.tensor([[4, 5], [6, 7]])
        kwargs = dict(value_memory=state.value_memory, value_residual=state.value_residual, copy_state=state.copy_state)
        decoder.prepare_cross_cache(state.memory, state.value_memory, state.value_residual)
        _, cache, _ = decoder(tokens, state.memory, state.memory_mask, state.source_bias, use_cache=True, **kwargs)
        rows = torch.tensor([1])
        cache.batch_select_indices(rows)
        decoder.select_cross_cache(rows)
        kwargs = dict(
            value_memory=state.value_memory[rows],
            value_residual=state.value_residual[rows],
            copy_state=state.copy_state.index_select(rows),
        )
        step, _, _ = decoder(
            torch.tensor([[8]]),
            state.memory[rows],
            state.memory_mask[rows],
            state.source_bias[rows],
            use_cache=True,
            past_key_values=cache,
            **kwargs,
        )
        decoder.clear_cross_cache()
        full, _, _ = decoder(
            torch.tensor([[6, 7, 8]]), state.memory[rows], state.memory_mask[rows], state.source_bias[rows], **kwargs
        )
        torch.testing.assert_close(step[:, -1], full[:, -1], atol=0.005 if dtype == "bfloat16" else 1e-6, rtol=1e-4)
        _, compact = generate_greedy(model, batch, tokenizer, 4, compact_finished=True)
        _, uncompact = generate_greedy(model, batch, tokenizer, 4, compact_finished=False)
        torch.testing.assert_close(compact, uncompact)
        generate_sampled(model, batch, tokenizer, 4, temperature=0.7, top_k=10, top_p=0.9)
        assert all(layer.cross._cache is None for layer in decoder.backbone.layers)
        save_checkpoint(tmp_path / "last.pt", model, None, config, epoch=1, step=2)
        restored = EviSeqAFMR(config).eval()
        load_checkpoint(tmp_path / "last.pt", restored, config=config)
        torch.testing.assert_close(_forward(model, batch).loss, _forward(restored, batch).loss, rtol=0, atol=0)


@pytest.mark.parametrize("change", [{"enabled": False}, {"max_relative_rms": 0.1}, {"stride": 1}, {"num_heads": 2}])
def test_context_checkpoint_rejects_graph_or_semantic_changes(change, tmp_path):
    config, model, _, _ = _setup()
    save_checkpoint(tmp_path / "last.pt", model, None, config, epoch=1, step=1)
    changed = copy.deepcopy(config)
    changed["architecture"]["contextual_value"].update(change)
    with pytest.raises(ValueError, match="architecture_spec"):
        load_checkpoint(tmp_path / "last.pt", EviSeqAFMR(changed), config=changed)


def test_context_checkpoint_rejects_previous_global_value_mechanism(tmp_path):
    config, model, _, _ = _setup()
    path = tmp_path / "old_global.pt"
    save_checkpoint(path, model, None, config, epoch=1, step=1)
    state = torch.load(path, weights_only=False)
    state["architecture_spec"]["contextual_value"]["mechanism"] = "region_attention_bounded_values"
    torch.save(state, path)
    with pytest.raises(ValueError, match="architecture_spec"):
        load_checkpoint(path, EviSeqAFMR(config), config=config)


@pytest.mark.parametrize(
    "change", [{"stride": 5}, {"dim": 15}, {"num_heads": 3}, {"max_relative_rms": 0}, {"enabled": "true"}]
)
def test_context_config_rejects_invalid_settings(change):
    config = _config()
    config["architecture"]["contextual_value"].update(change)
    with pytest.raises(ValueError, match="contextual_value"):
        validate_config(config)


def test_direct_projection_removes_entire_context_branch_and_legacy_spec_is_stable():
    config = _config()
    config["architecture"]["bridge_mode"] = "direct_projection"
    validate_config(config)
    _, model, batch, _ = _setup(config)
    assert not hasattr(model.bridge, "contextual_value")
    assert not any("contextual_value" in name for name, _ in model.named_parameters())
    assert _forward(model, batch).bridge.value_residual is None
    assert "contextual_value" not in architecture_spec(config)
    config["architecture"].pop("bridge_mode")
    config["architecture"]["contextual_value"]["enabled"] = False
    disabled = architecture_spec(config)
    config["architecture"].pop("contextual_value")
    assert architecture_spec(config) == disabled


def test_generation_compacts_value_residual_when_a_row_finishes(monkeypatch):
    _, model, batch, tokenizer = _setup()
    torch.nn.init.normal_(model.bridge.contextual_value.output.weight, std=0.2)
    original_forward = model.decoder.forward
    observed = []

    def force_first_row_eos(*args, **kwargs):
        observed.append(kwargs["value_residual"].clone())
        logits, cache, loss = original_forward(*args, **kwargs)
        logits = logits.clone()
        logits[:, -1, tokenizer.eos_token_id] = -10000
        if kwargs.get("past_key_values") is None:
            logits[0, -1, tokenizer.eos_token_id] = 10000
        return logits, cache, loss

    monkeypatch.setattr(model.decoder, "forward", force_first_row_eos)
    _, compact = generate_greedy(model, batch, tokenizer, 4, compact_finished=True)
    assert [value.shape[0] for value in observed] == [2, 1, 1, 1]
    for value in observed[1:]:
        torch.testing.assert_close(value, observed[0][1:], rtol=0, atol=0)
    _, uncompact = generate_greedy(model, batch, tokenizer, 4, compact_finished=False)
    torch.testing.assert_close(compact, uncompact)


@pytest.mark.parametrize("script", ["run_pubmed_pair.sh", "run_arxiv.sh"])
@pytest.mark.parametrize("mode,context", [("afmr", "false"), ("direct_projection", "false")])
def test_runner_materialization_selects_full_and_ablation_graphs(script, mode, context, tmp_path):
    root = Path(__file__).parents[1]
    source = (root / "scripts" / script).read_text()
    body = re.search(r"<<'PY'\n(.*?)\nPY", source, flags=re.S).group(1)
    template = root / "configs" / ("afmr_arxiv.yaml" if script == "run_arxiv.sh" else "afmr_pubmed.yaml")
    config = load_config(template)
    # A previously edited ablation template must not silently override full mode.
    config["architecture"]["bridge_mode"] = "direct_projection"
    config["training"]["salience_loss_weight"] = 0.0
    config.pop("_meta")
    import yaml

    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump(config))
    destination = tmp_path / "resolved.yaml"
    args = [str(template), str(destination), "/local/encoder", "/local/decoder", str(tmp_path), "/local/data"]
    if script == "run_pubmed_pair.sh":
        args += ["afmr_hyper_operator", mode, "true", context, "0"]
    else:
        args += ["8", "6", "4", "0", "0", "1", "4", "8192", "512", "8", "512", "32", mode, "true", context]
    subprocess.run([sys.executable, "-c", body, *args], check=True, capture_output=True, text=True)
    result = load_config(destination)
    assert result["architecture"].get("bridge_mode", "afmr") == mode
    assert contextual_value_settings(result["architecture"])["enabled"] is (context == "true" and mode == "afmr")
    assert hyper_operator_settings(result["architecture"])["enabled"] is (mode == "afmr")
    assert result["decoder"]["grounded_copy"]["enabled"] is True
    for section, keys in (
        (
            "training",
            ("full_encoder_lr", "full_decoder_lr", "full_bridge_lr", "full_cross_attention_lr", "max_grad_norm"),
        ),
        ("generation", ("temperature", "top_k", "top_p", "do_sample")),
    ):
        assert {k: result[section][k] for k in keys} == {k: config[section][k] for k in keys}


def _ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(8 + rank)
        config, model, batch, _ = _setup()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, broadcast_buffers=False)
        # Ranks see different source/target examples, with copy and checkpointing active.
        batch = {k: v[rank : rank + 1] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        for stage in ("interface_warmup", "full_finetune"):
            set_stage_trainability(model, stage)
            optimizer = build_optimizer(model, config, stage)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                _forward(model, batch).loss.backward()
                optimizer.step()
            for parameter in model.module.bridge.contextual_value.parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                peers = [torch.empty_like(parameter) for _ in range(2)]
                dist.all_gather(peers, parameter.detach())
                torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_contextual_value_two_rank_ddp_updates_remain_synchronized(tmp_path):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_ddp_worker, args=(rendezvous,), nprocs=2, join=True)
