from types import SimpleNamespace

import torch
import torch.nn as nn
from eviseq_afmr.modeling.decoder import CopiedCrossAttention


def _attention():
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
    return CopiedCrossAttention(self_attention, norm, config, 0.0)


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
