from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from eviseq_afmr.modeling.decoder import CopiedCrossAttention
from eviseq_afmr.modeling.region_query import RegionQueryRead, pool_regions


def test_pool_regions_shapes_masks_content_and_separates_key_value_sources():
    key = torch.arange(2 * 7 * 4, dtype=torch.float32).reshape(2, 7, 4)
    value = key + 1000.0
    content = torch.tensor([[False, True, True, True, True, False, False], [False] * 7], dtype=torch.bool)

    region_keys, region_values, region_mask = pool_regions(key, value, content, window_size=2, stride=1)

    assert region_keys.shape == (2, 7, 4)
    assert region_values.shape == region_keys.shape
    assert region_mask.shape == (2, 7)
    assert region_mask[0].tolist() == [True, True, True, False, False, False, False]
    assert not region_mask[1].any()
    offset = torch.where(region_mask[0, :, None], 1000.0, 0.0).expand_as(region_keys[0])
    torch.testing.assert_close(region_values[0] - region_keys[0], offset)
    assert region_keys[1].count_nonzero() == 0
    assert region_values[1].count_nonzero() == 0


def test_query_read_zero_output_and_all_invalid_region_is_safe():
    torch.manual_seed(7)
    module = RegionQueryRead(
        hidden_size=12,
        rank=8,
        gate_init=0.05,
        gate_max=0.20,
        num_heads=2,
        num_query_heads=3,
        query_head_dim=5,
    ).eval()
    query = torch.randn(2, 3, 12)
    q_base = torch.randn(2, 3, 3, 5)
    keys = torch.randn(2, 4, 12)
    values = torch.randn(2, 4, 12)
    mask = torch.tensor([[True, True, False, False], [False, False, False, False]])

    output = module(query, q_base, keys, values, mask)

    assert output.shape == q_base.shape  # projected Q width 15 != hidden width 12
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    assert torch.isfinite(output).all()


def test_query_read_first_and_second_step_gradients_and_cache_parity():
    torch.manual_seed(11)
    module = RegionQueryRead(
        hidden_size=12,
        rank=8,
        gate_init=0.05,
        gate_max=0.20,
        num_heads=2,
        num_query_heads=3,
        query_head_dim=5,
    )
    query = torch.randn(2, 3, 12)
    q_base = torch.randn(2, 3, 3, 5)
    keys = torch.randn(2, 5, 12)
    values = torch.randn(2, 5, 12)
    mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])
    optimizer = torch.optim.AdamW(module.parameters(), lr=1.0e-2)

    optimizer.zero_grad(set_to_none=True)
    first = module(query, q_base, keys, values, mask)
    # A non-zero upstream signal models the decoder loss.  Squaring the exact
    # zero output would have a zero derivative and would not test the route.
    first.sum().backward()
    assert module.out_proj.weight.grad is not None and module.out_proj.weight.grad.abs().sum() > 0
    assert module.q_proj.weight.grad is not None and module.q_proj.weight.grad.abs().sum() == 0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second = module(query, q_base, keys, values, mask)
    second.sum().backward()
    assert module.out_proj.weight.grad is not None and module.out_proj.weight.grad.abs().sum() > 0
    assert module.q_proj.weight.grad is not None and module.q_proj.weight.grad.abs().sum() > 0
    assert module.k_proj.weight.grad is not None and module.k_proj.weight.grad.abs().sum() > 0
    assert module.v_proj.weight.grad is not None and module.v_proj.weight.grad.abs().sum() > 0

    module.eval()
    with torch.no_grad():
        expected = module(query, q_base, keys, values, mask)
        module.prepare_cache(keys, values, mask)
        cached = module(query, q_base)
    torch.testing.assert_close(cached, expected, rtol=1e-6, atol=1e-6)
    module.select_cache(torch.tensor([1]))
    with torch.no_grad():
        compact = module(query[1:], q_base[1:])
    torch.testing.assert_close(compact, cached[1:], rtol=1e-6, atol=1e-6)
    module.clear_cache()


def test_query_read_caps_actual_projected_query_perturbation_per_head():
    torch.manual_seed(19)
    module = RegionQueryRead(
        hidden_size=12,
        rank=8,
        gate_init=0.05,
        gate_max=0.20,
        num_query_heads=3,
        query_head_dim=5,
    )
    with torch.no_grad():
        module.out_proj.weight.fill_(100.0)
    query = torch.randn(2, 3, 12)
    q_base = torch.randn(2, 3, 3, 5)
    keys = torch.randn(2, 5, 12)
    values = torch.randn(2, 5, 12)
    mask = torch.ones(2, 5, dtype=torch.bool)

    delta = module(query, q_base, keys, values, mask)
    delta_rms = delta.float().square().mean(-1).sqrt()
    query_rms = q_base.float().square().mean(-1).sqrt()
    assert torch.all(delta_rms <= 0.20 * query_rms + 1.0e-6)
    assert torch.isfinite(delta).all()


def test_copied_cross_attention_accepts_non_square_projected_query():
    torch.manual_seed(23)
    config = SimpleNamespace(hidden_size=12, num_attention_heads=3, num_key_value_heads=1, head_dim=5)
    attention = nn.Module()
    attention.q_proj = nn.Linear(12, 15, bias=False)
    attention.k_proj = nn.Linear(12, 5, bias=False)
    attention.v_proj = nn.Linear(12, 5, bias=False)
    attention.o_proj = nn.Linear(15, 12, bias=False)
    attention.q_norm = nn.Identity()
    attention.k_norm = nn.Identity()
    input_norm = nn.LayerNorm(12)
    region_config = {"enabled": True, "rank": 8, "gate_init": 0.05, "gate_max": 0.20}
    with_region = CopiedCrossAttention(attention, input_norm, config, 0.0, region_query_config=region_config).eval()
    without_region = CopiedCrossAttention(attention, input_norm, config, 0.0).eval()
    query = torch.randn(2, 3, 12)
    memory = torch.randn(2, 4, 12)
    mask = torch.ones(2, 4, dtype=torch.bool)
    bias = torch.zeros(2, 4)
    with torch.no_grad():
        expected = without_region(query, memory, mask, bias)
        actual = with_region(query, memory, mask, bias, region_keys=memory, region_values=memory, region_mask=mask)
    assert actual.shape == query.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
