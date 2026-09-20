import copy
from pathlib import Path

import torch

from eviseq_afmr.config import load_config, validate_config
from eviseq_afmr.modeling.exact_routing import (
    ExactRoutingState,
    ExactRoutingBridge,
    ExactTokenRouter,
    _masked_softmax,
)
from eviseq_afmr.modeling.outputs import EncoderState
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.training.checkpoint import architecture_spec
from eviseq_afmr.training.optimizer import build_optimizer, set_stage_trainability


def test_masked_softmax_has_zero_mass_for_padding_and_empty_rows():
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [False, False]])
    probabilities = _masked_softmax(logits, mask, dim=-1)
    torch.testing.assert_close(probabilities, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))


def test_blocks_are_deterministic_and_non_overlapping():
    router = ExactTokenRouter(hidden_size=4, rank=2, block_size=2, query_chunk_size=1)
    memory = torch.arange(20, dtype=torch.float32).view(1, 5, 4)
    state = router.build(memory, torch.tensor([[True, True, False, True, True]]))
    assert state.block_keys.shape[:2] == (1, 3)
    assert state.block_mask.tolist() == [[True, True, True]]
    assert state.token_mask.tolist() == [[True, True, False, True, True]]
    assert state.block_size == 2


def test_random_route_logits_start_at_attention_scale():
    torch.manual_seed(2)
    router = ExactTokenRouter(hidden_size=1024, rank=128)
    query = torch.randn(256, 1024)
    keys = torch.randn(256, 1024)
    logits = (router.query_proj(query) * router.key_proj(keys)).sum(-1) / (router.rank**0.5)
    assert 0.5 < float(logits.std().detach()) < 1.5


def test_short_final_block_has_no_uniform_logit_preference():
    router = ExactTokenRouter(hidden_size=2, rank=2, block_size=128)
    with torch.no_grad():
        router.query_proj.weight.zero_()
        router.key_proj.weight.zero_()
    memory = torch.zeros(1, 129, 2)
    memory[0, -1, 0] = 1.0
    state = router.build(memory, torch.ones(1, 129, dtype=torch.bool))
    output = router(torch.zeros(1, 1, 2), state)
    torch.testing.assert_close(output[0, 0, 0], torch.tensor(1.0 / 129), rtol=1.0e-6, atol=1.0e-7)


def test_exact_route_reads_original_token_values_and_depends_on_query():
    router = ExactTokenRouter(hidden_size=2, rank=2, block_size=2, query_chunk_size=8)
    with torch.no_grad():
        router.query_proj.weight.copy_(torch.eye(2))
        router.key_proj.weight.copy_(torch.eye(2))
    values = torch.tensor([[[10.0, 0.0], [0.0, 20.0], [30.0, 0.0], [0.0, 40.0]]])
    keys = torch.tensor([[[8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]]])
    state = ExactRoutingState(
        token_keys=keys,
        block_keys=torch.tensor([[[4.0, 4.0], [4.0, 4.0]]]),
        values=values,
        token_mask=torch.ones(1, 4, dtype=torch.bool),
        block_mask=torch.ones(1, 2, dtype=torch.bool),
        block_size=2,
    )
    first = router(torch.tensor([[[8.0, 0.0]]]), state)
    second = router(torch.tensor([[[0.0, 8.0]]]), state)
    assert first[0, 0, 0] > first[0, 0, 1]
    assert second[0, 0, 1] > second[0, 0, 0]
    assert not torch.allclose(first, second)


def test_route_gradients_reach_query_and_key_projections():
    torch.manual_seed(3)
    router = ExactTokenRouter(hidden_size=6, rank=3, block_size=2, query_chunk_size=2)
    memory = torch.randn(2, 5, 6, requires_grad=True)
    query = torch.randn(2, 4, 6, requires_grad=True)
    state = router.build(memory, torch.ones(2, 5, dtype=torch.bool))
    loss = router(query, state).square().mean()
    loss.backward()
    assert router.query_proj.weight.grad is not None
    assert router.key_proj.weight.grad is not None
    assert memory.grad is not None


def test_bridge_keeps_copy_anchor_and_no_static_source_bias():
    router = ExactTokenRouter(hidden_size=4, rank=2, block_size=2, query_chunk_size=2)
    bridge = ExactRoutingBridge(4, 4, {"bridge_mode": "afmr"}, router)
    final = torch.randn(1, 5, 4)
    state = EncoderState(
        final,
        (),
        torch.ones(1, 5, dtype=torch.bool),
        torch.tensor([[True, True, True, False, False]]),
    )
    output = bridge(state, torch.empty(1, 0, 4), torch.empty(1, 0, dtype=torch.bool), torch.ones(1))
    torch.testing.assert_close(output.memory, final)
    torch.testing.assert_close(output.exact_route.values, output.memory)
    assert output.source_bias.abs().sum() == 0


def test_cached_row_selection_matches_full_route():
    torch.manual_seed(5)
    router = ExactTokenRouter(hidden_size=4, rank=2, block_size=2, query_chunk_size=2)
    memory = torch.randn(3, 5, 4)
    state = router.build(memory, torch.ones(3, 5, dtype=torch.bool))
    query = torch.randn(3, 3, 4)
    selected = torch.tensor([2, 0])
    torch.testing.assert_close(
        router(query.index_select(0, selected), state.index_select(selected)),
        router(query, state).index_select(0, selected),
    )


def _tiny_exact_config() -> dict:
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["architecture"]["name"] = "afmr_exact_routing"
    config["architecture"]["bridge_mode"] = "afmr"
    config["architecture"]["exact_routing"] = {
        "enabled": True,
        "rank": 4,
        "block_size": 2,
        "query_chunk_size": 2,
        "gate_init": 0.0,
        "gate_max": 0.2,
    }
    config["model"]["gradient_checkpointing"] = False
    validate_config(config)
    return config


def _tiny_batch() -> dict[str, torch.Tensor]:
    source = torch.tensor([[1, 7, 11, 13, 2], [1, 17, 19, 23, 2]])
    target = torch.tensor([[1, 29, 31, 2], [1, 37, 41, 2]])
    return {
        "input_ids": source,
        "attention_mask": torch.ones_like(source, dtype=torch.bool),
        "source_content_mask": torch.tensor([[False, True, True, True, False]] * 2),
        "decoder_prompt_ids": torch.tensor([[1, 5]] * 2),
        "decoder_prompt_mask": torch.ones(2, 2, dtype=torch.bool),
        "decoder_input_ids": target,
        "decoder_attention_mask": torch.ones_like(target, dtype=torch.bool),
        "labels": target,
    }


def test_full_exact_graph_initializes_like_direct_projection_and_has_trainable_warmup_route():
    full_config = _tiny_exact_config()
    direct_config = copy.deepcopy(full_config)
    direct_config["architecture"]["bridge_mode"] = "direct_projection"
    validate_config(direct_config)
    torch.manual_seed(211)
    full = EviSeqAFMR(full_config).eval()
    torch.manual_seed(211)
    direct = EviSeqAFMR(direct_config).eval()
    batch = _tiny_batch()
    with torch.no_grad():
        full_output = full(**batch)
        direct_output = direct(**batch)
    torch.testing.assert_close(full_output.logits, direct_output.logits, rtol=0, atol=0)
    assert architecture_spec(full_config)["exact_routing"]["enabled"] is True
    assert architecture_spec(direct_config)["exact_routing"]["enabled"] is False

    full.train()
    set_stage_trainability(full, "interface_warmup")
    optimizer = build_optimizer(full, full_config, "interface_warmup")
    routed = [(name, parameter) for name, parameter in full.named_parameters() if "exact_route" in name]
    assert routed and all(parameter.requires_grad for _, parameter in routed)
    optimizer.zero_grad(set_to_none=True)
    loss = full(**batch).loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    gates = [parameter for name, parameter in routed if name.endswith("exact_route_gate_raw")]
    assert gates and any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in gates)
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    loss = full(**batch).loss
    assert loss is not None
    loss.backward()
    router = full.decoder.exact_router
    assert router is not None
    assert router.query_proj.weight.grad is not None
    assert torch.isfinite(router.query_proj.weight.grad).all()
    assert router.query_proj.weight.grad.abs().sum() > 0
