from pathlib import Path

import torch

from eviseq_afmr.config import load_config
from eviseq_afmr.data.copy_alignment import COPY_INPUT_KEYS
from eviseq_afmr.evaluation.generate import generate_greedy
from eviseq_afmr.modeling.evidence_route import pool_source_regions
from eviseq_afmr.modeling.grounded_copy import CopyState, GroundedCopyHead
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import build_loaders
from eviseq_afmr.training.optimizer import build_optimizer, set_stage_trainability


def _tiny_run():
    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    config["decoder"]["grounded_copy"]["enabled"] = True
    config["decoder"]["evidence_router"].update(enabled=True, region_width=4, last_n_layers=1)
    config["decoder"]["ce_chunk_size"] = 8
    model = EviSeqAFMR(config)
    loader = build_loaders(config, max_train_examples=2)["train"]
    return config, model, next(iter(loader)), loader.collate_fn.decoder_tokenizer


def test_regions_exclude_prompt_and_keep_distinct_source_occurrences():
    memory = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2).requires_grad_()
    content = torch.tensor([[False, True, True, True, True, False]])
    regions, ids = pool_source_regions(memory, content, 2)
    assert ids.tolist() == [[-1, 0, 0, 1, 1, -1]]
    torch.testing.assert_close(regions[0, 0], memory[0, 1:3].mean(0))
    torch.testing.assert_close(regions[0, 1], memory[0, 3:5].mean(0))
    regions.sum().backward()
    assert memory.grad[0, 0].eq(0).all()
    assert memory.grad[0, 2].abs().sum() > 0


def test_router_does_not_change_initial_shared_weights():
    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    config["decoder"]["grounded_copy"]["enabled"] = True
    torch.manual_seed(23)
    baseline = EviSeqAFMR(config)
    config["decoder"]["evidence_router"].update(enabled=True, region_width=4, last_n_layers=1)
    torch.manual_seed(23)
    routed = EviSeqAFMR(config)
    for name, weight in baseline.state_dict().items():
        torch.testing.assert_close(weight, routed.state_dict()[name], rtol=0, atol=0)
    routed_cross = routed.decoder.backbone.layers[-1].cross
    torch.testing.assert_close(routed_cross.region_key_proj.weight, routed_cross.k_proj.weight, rtol=0, atol=0)


def test_copy_route_uses_source_position_before_vocabulary_marginalization():
    torch.manual_seed(3)
    head = GroundedCopyHead(8, 4, 0.1, route_init=0.2, route_max=0.5)
    state = CopyState(
        torch.zeros(1, 3, 4),
        torch.tensor([[4, 4, 5]]),
        torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, 3),
        torch.tensor([[0, 1, 1]]),
    )
    hidden = torch.zeros(1, 1, 8)
    left = head.distribution(hidden, state, torch.tensor([[[4.0, -4.0]]]))[0].exp()
    right = head.distribution(hidden, state, torch.tensor([[[-4.0, 4.0]]]))[0].exp()
    assert left[0, 0, 0] > right[0, 0, 0]
    assert left[0, 0, 1] < right[0, 0, 1]


def test_alignment_maps_decoder_occurrences_to_dominant_encoder_region():
    head = GroundedCopyHead(8, 4, 0.1, route_init=0.2, route_max=0.5)
    state = head.prepare(
        torch.randn(1, 4, 8),
        torch.zeros(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
        torch.nn.Embedding(12, 8),
        copy_token_ids=torch.tensor([[4, 4]]),
        copy_token_mask=torch.ones(1, 2, dtype=torch.bool),
        copy_encoder_indices=torch.tensor([[0, 1, 2, 3]]),
        copy_token_indices=torch.tensor([[0, 0, 1, 1]]),
        copy_alignment_weights=torch.tensor([[0.7, 0.3, 0.2, 0.8]]),
        source_region_ids=torch.tensor([[0, 1, 0, 1]]),
    )
    assert state.region_ids.tolist() == [[0, 1]]


def test_router_full_chunk_loss_gradient_and_optimizer_update():
    torch.manual_seed(6)
    config, model, batch, _ = _tiny_run()
    model.train()
    inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    dense = model(**inputs, return_logits=True)
    chunked = model(**inputs, return_logits=False)
    torch.testing.assert_close(dense.loss, chunked.loss, atol=1e-5, rtol=1e-5)
    assert dense.bridge.region_states is not None
    assert dense.bridge.copy_state.region_ids is not None
    assert torch.equal(
        dense.bridge.copy_state.region_ids[dense.bridge.copy_state.mask].ge(0),
        torch.ones_like(dense.bridge.copy_state.region_ids[dense.bridge.copy_state.mask], dtype=torch.bool),
    )
    set_stage_trainability(model, "interface_warmup")
    optimizer = build_optimizer(model, config, "interface_warmup")
    last_cross = model.decoder.backbone.layers[-1].cross
    copy_head = model.decoder.grounded_copy
    before = last_cross.region_key_proj.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    model(**inputs, return_logits=False).loss.backward()
    for parameter in (
        last_cross.region_key_proj.weight,
        last_cross.route_strength_raw,
        copy_head.route_strength_raw,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(last_cross.region_key_proj.weight, before)


def test_zero_router_strength_recovers_afmr_copy_distribution():
    torch.manual_seed(11)
    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    config["decoder"]["grounded_copy"]["enabled"] = True
    loader = build_loaders(config, max_train_examples=2)["train"]
    batch = next(iter(loader))
    baseline = EviSeqAFMR(config).eval()
    config["decoder"]["evidence_router"].update(enabled=True, region_width=4, last_n_layers=1)
    routed = EviSeqAFMR(config).eval()
    shared = {name: value for name, value in baseline.state_dict().items() if name in routed.state_dict()}
    routed.load_state_dict(shared, strict=False)
    with torch.no_grad():
        routed.decoder.backbone.layers[-1].cross.route_strength_raw.fill_(-1000.0)
        routed.decoder.grounded_copy.route_strength_raw.fill_(-1000.0)
        inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
        baseline_logits = baseline(**inputs, return_logits=True).logits
        routed_logits = routed(**inputs, return_logits=True).logits
    torch.testing.assert_close(routed_logits, baseline_logits, atol=2e-6, rtol=2e-6)


def test_routed_cache_and_compaction_match_full_decode():
    torch.manual_seed(7)
    _, model, batch, tokenizer = _tiny_run()
    model.eval()
    with torch.no_grad():
        state = model.encode_source(
            batch["input_ids"],
            batch["attention_mask"],
            batch["source_content_mask"],
            batch["decoder_prompt_ids"],
            batch["decoder_prompt_mask"],
            torch.full((2,), 16.0),
            **{key: batch[key] for key in COPY_INPUT_KEYS},
        )
        decoder = model.decoder
        tokens = torch.tensor([[4, 5], [6, 7]])
        kwargs = dict(
            value_memory=state.value_memory,
            copy_state=state.copy_state,
            region_states=state.region_states,
            source_region_ids=state.source_region_ids,
        )
        decoder.prepare_cross_cache(state.memory, state.value_memory, state.region_states, state.source_region_ids)
        _, cache, _ = decoder(tokens, state.memory, state.memory_mask, state.source_bias, use_cache=True, **kwargs)
        rows = torch.tensor([1])
        cache.batch_select_indices(rows)
        decoder.select_cross_cache(rows)
        selected = {
            key: value.index_select(0, rows) if isinstance(value, torch.Tensor) else value.index_select(rows)
            for key, value in kwargs.items()
        }
        cached, _, _ = decoder(
            torch.tensor([[8]]),
            state.memory[rows],
            state.memory_mask[rows],
            state.source_bias[rows],
            past_key_values=cache,
            use_cache=True,
            **selected,
        )
        decoder.clear_cross_cache()
        full, _, _ = decoder(
            torch.tensor([[6, 7, 8]]),
            state.memory[rows],
            state.memory_mask[rows],
            state.source_bias[rows],
            **selected,
        )
        torch.testing.assert_close(cached[:, -1], full[:, -1], atol=1e-5, rtol=1e-5)
        _, compact = generate_greedy(model, batch, tokenizer, 4, compact_finished=True)
        _, uncompact = generate_greedy(model, batch, tokenizer, 4, compact_finished=False)
        torch.testing.assert_close(compact, uncompact)


def test_bf16_autocast_keeps_router_loss_finite_and_updates_encoder():
    torch.manual_seed(9)
    config, model, batch, _ = _tiny_run()
    set_stage_trainability(model, "full_finetune")
    inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(**inputs, return_logits=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder.model.embed_tokens.weight.grad is not None
    assert model.encoder.model.embed_tokens.weight.grad.abs().sum() > 0
    assert model.decoder.backbone.layers[-1].cross.region_key_proj.weight.grad.abs().sum() > 0


def test_bf16_weight_evaluation_uses_routed_cross_cache():
    torch.manual_seed(10)
    config, _, batch, tokenizer = _tiny_run()
    config["model"]["dtype"] = "bfloat16"
    model = EviSeqAFMR(config).eval()
    with torch.no_grad():
        texts, tokens = generate_greedy(model, batch, tokenizer, 3)
    assert len(texts) == 2
    assert tokens.shape[0] == 2
    assert all(layer.cross._cache is None for layer in model.decoder.backbone.layers)
