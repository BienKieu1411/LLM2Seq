from types import SimpleNamespace

import torch
import torch.nn as nn
from grounded_summary.modeling.decoder import CopiedCrossAttention


def _attention(region_router_config=None):
    self_attention = SimpleNamespace(
        q_proj=nn.Linear(8, 8, bias=False),
        k_proj=nn.Linear(8, 4, bias=False),
        v_proj=nn.Linear(8, 4, bias=False),
        o_proj=nn.Linear(8, 8, bias=False),
        q_norm=nn.Identity(),
        k_norm=nn.Identity(),
    )
    norm = nn.Identity()
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=4)
    return CopiedCrossAttention(self_attention, norm, config, 0.0, region_router_config)


def _regions(batch=2, source_length=7, hidden=8):
    region_memory = torch.randn(batch, 2, hidden)
    region_mask = torch.ones(batch, 2, dtype=torch.bool)
    region_index = torch.arange(source_length).remainder(2).repeat(batch, 1)
    region_token_mask = torch.ones(batch, source_length, dtype=torch.bool)
    return region_memory, region_mask, region_index, region_token_mask


def test_cross_attention_bias_and_cache_contract():
    attention = _attention().eval()
    query = torch.randn(2, 3, 8)
    memory = torch.randn(2, 7, 8)
    mask = torch.ones(2, 7, dtype=torch.bool)
    bias = torch.zeros(2, 7)
    bias[:, 4:] = -2.0
    output = attention(query, memory, mask, bias)
    assert output.shape == query.shape
    attention.prepare_cache(memory)
    cached = attention(query, memory, mask, bias)
    assert torch.allclose(output, cached, atol=1.0e-6)
    attention.clear_cache()


def test_padding_is_masked_even_without_source_bias():
    torch.manual_seed(4)
    attention = _attention().eval()
    query = torch.randn(1, 3, 8)
    memory = torch.randn(1, 7, 8)
    mask = torch.tensor([[True, True, True, False, False, False, False]])
    expected = attention(query, memory[:, :3], mask[:, :3], None)
    padded = attention(query, memory, mask, None)
    torch.testing.assert_close(padded, expected)


def test_bfloat16_source_bias_backward_is_live():
    attention = _attention().to(dtype=torch.bfloat16)
    query = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    memory = torch.randn(2, 9, 8, dtype=torch.bfloat16, requires_grad=True)
    bias = torch.zeros(2, 9, requires_grad=True)
    mask = torch.ones(2, 9, dtype=torch.bool)
    output = attention(query, memory, mask, bias)
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    for tensor in (query, memory, bias):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_zero_gate_region_router_preserves_cross_attention_output():
    torch.manual_seed(13)
    baseline = _attention().eval()
    routed = _attention({"enabled": True, "rank": 4, "gate_max": 0.10, "temperature": 1.0}).eval()
    routed.load_state_dict(baseline.state_dict(), strict=False)
    query = torch.randn(2, 3, 8)
    memory = torch.randn(2, 7, 8)
    mask = torch.ones(2, 7, dtype=torch.bool)
    bias = torch.randn(2, 7)
    regions = _regions()
    with torch.no_grad():
        expected = baseline(query, memory, mask, bias)
        actual = routed(
            query,
            memory,
            mask,
            bias,
            region_memory=regions[0],
            region_mask=regions[1],
            region_index=regions[2],
            region_token_mask=regions[3],
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_region_router_bias_is_query_conditioned_bounded_and_masked():
    torch.manual_seed(14)
    attention = _attention({"enabled": True, "rank": 4, "gate_max": 0.10, "temperature": 1.0}).eval()
    with torch.no_grad():
        attention.region_router_gate.fill_(1.0)
        attention.region_router_query.weight.normal_()
        attention.region_router_key.weight.normal_()
    query = torch.randn(2, 3, 8)
    query[1] = query[0] * -1
    regions = _regions()
    regions = (*regions[:3], regions[3].clone())
    regions[3][:, 0] = False
    first = attention._region_bias(query, *regions)
    second = attention._region_bias(query[[1, 0]], *regions)
    assert first is not None and first.shape == (2, 3, 7)
    assert torch.isfinite(first).all()
    assert first.abs().max() <= 0.10 + 1.0e-6
    assert torch.all(first[..., 0] == 0)
    assert not torch.allclose(first[0], second[1])


def test_region_router_gradients_open_after_gate_update():
    torch.manual_seed(15)
    attention = _attention({"enabled": True, "rank": 4, "gate_max": 0.10, "temperature": 1.0})
    query = torch.randn(2, 3, 8, requires_grad=True)
    memory = torch.randn(2, 7, 8, requires_grad=True)
    regions = _regions()
    regions = (regions[0].requires_grad_(), *regions[1:])
    mask = torch.ones(2, 7, dtype=torch.bool)
    bias = torch.zeros(2, 7)
    first = attention(
        query,
        memory,
        mask,
        bias,
        region_memory=regions[0],
        region_mask=regions[1],
        region_index=regions[2],
        region_token_mask=regions[3],
    )
    first.square().mean().backward()
    assert attention.region_router_gate.grad is not None
    assert torch.isfinite(attention.region_router_gate.grad)
    assert attention.region_router_gate.grad.abs() > 0
    with torch.no_grad():
        attention.region_router_gate.add_(0.5)
    attention.zero_grad(set_to_none=True)
    query.grad = None
    memory.grad = None
    regions[0].grad = None
    second = attention(
        query,
        memory,
        mask,
        bias,
        region_memory=regions[0],
        region_mask=regions[1],
        region_index=regions[2],
        region_token_mask=regions[3],
    )
    second.square().mean().backward()
    for parameter in (attention.region_router_query.weight, attention.region_router_key.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert regions[0].grad is not None and torch.isfinite(regions[0].grad).all()


def test_region_router_key_cache_matches_projection_and_compacts():
    torch.manual_seed(16)
    attention = _attention({"enabled": True, "rank": 4, "gate_max": 0.10, "temperature": 1.0}).eval()
    query = torch.randn(2, 3, 8)
    memory = torch.randn(2, 7, 8)
    mask = torch.ones(2, 7, dtype=torch.bool)
    bias = torch.zeros(2, 7)
    regions = _regions()
    with torch.no_grad():
        attention.region_router_gate.fill_(1.0)
        expected = attention._region_bias(query, *regions)
        attention.prepare_cache(memory, region_memory=regions[0])
        assert attention._region_key_cache is not None
        cached = attention._region_bias(query, *regions, attention._region_key_cache)
        attention._region_key_cache = attention._region_key_cache.index_select(0, torch.tensor([1]))
        assert attention._region_key_cache.shape[0] == 1
        attention.clear_cache()
    torch.testing.assert_close(cached, expected, rtol=0, atol=0)
    assert attention._region_key_cache is None


def test_region_router_empty_content_is_neutral():
    attention = _attention({"enabled": True, "rank": 4, "gate_max": 0.10, "temperature": 1.0}).eval()
    region_memory = torch.zeros(1, 2, 8)
    region_mask = torch.zeros(1, 2, dtype=torch.bool)
    region_index = torch.zeros(1, 5, dtype=torch.long)
    region_token_mask = torch.zeros(1, 5, dtype=torch.bool)
    bias = attention._region_bias(torch.randn(1, 3, 8), region_memory, region_mask, region_index, region_token_mask)
    assert bias is not None and torch.isfinite(bias).all() and bias.eq(0).all()


def test_selected_decoder_cache_matches_surviving_rows():
    from pathlib import Path

    from grounded_summary.config import load_config
    from grounded_summary.modeling.model import AFMRModel

    cfg = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    decoder = AFMRModel(cfg).decoder.eval()
    memory = torch.randn(3, 7, 24)
    mask = torch.ones(3, 7, dtype=torch.bool)
    bias = torch.randn(3, 7)
    tokens = torch.tensor([[3, 4], [5, 6], [7, 8]])
    rows = torch.tensor([0, 2])
    with torch.no_grad():
        decoder.prepare_cross_cache(memory)
        _, cache, _ = decoder(tokens, memory, mask, bias, use_cache=True)
        cache.batch_select_indices(rows)
        decoder.select_cross_cache(rows)
        next_ids = torch.tensor([[9], [10]])
        cached, _, _ = decoder(next_ids, memory[rows], mask[rows], bias[rows], past_key_values=cache, use_cache=True)
        decoder.clear_cross_cache()
        full, _, _ = decoder(torch.cat((tokens[rows], next_ids), dim=1), memory[rows], mask[rows], bias[rows])
    torch.testing.assert_close(cached[:, -1], full[:, -1], atol=1e-6, rtol=1e-5)
