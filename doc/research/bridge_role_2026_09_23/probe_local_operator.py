"""Synthetic mechanism checks only; no trained model or ROUGE claims.

Run with bienkieu_env and PYTHONPATH=src/eviseq_new. No model downloads.
"""

import json
from pathlib import Path

import torch
from eviseq_afmr.modeling.decoder import CopiedCrossAttention
from transformers import Qwen3Config, Qwen3Model


def local_difference(h, content, left_weight, right_weight):
    pair = (content[:, 1:] & content[:, :-1]).unsqueeze(-1)
    left = torch.zeros_like(h)
    right = torch.zeros_like(h)
    left[:, 1:] = (h[:, :-1] - h[:, 1:]) * pair
    right[:, :-1] = (h[:, 1:] - h[:, :-1]) * pair
    return h + left * left_weight + right * right_weight


def main():
    torch.manual_seed(42)
    cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=6,
        max_position_embeddings=32,
    )
    base = Qwen3Model(cfg).double()
    layer = base.layers[0]
    cross = CopiedCrossAttention(layer.self_attn, layer.input_layernorm, cfg, 0.0).double()
    cross.eval()
    h = torch.randn(2, 7, 24, dtype=torch.float64)
    q = torch.randn(2, 3, 24, dtype=torch.float64)
    visible = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    content = visible.clone()
    content[:, 0] = False
    a = torch.zeros(24, dtype=torch.float64, requires_grad=True)
    b = torch.zeros_like(a, requires_grad=True)
    memory = local_difference(h, content, a, b)
    initial_error = (memory - h).abs().max().item()
    reference = cross(q, h, visible, None, value_memory=h)
    output = cross(q, memory, visible, None, value_memory=h)
    initial_cross_error = (reference - output).abs().max().item()
    (output * torch.randn_like(output)).sum().backward()
    grad_a, grad_b = a.grad.norm().item(), b.grad.norm().item()
    assert initial_error == initial_cross_error == 0.0
    assert grad_a > 0 and grad_b > 0

    a_probe = torch.linspace(-0.15, 0.1, 24, dtype=h.dtype)
    b_probe = torch.linspace(0.2, -0.05, 24, dtype=h.dtype)
    adjusted = local_difference(h, content, a_probe, b_probe)
    key0, value0 = cross._memory_kv(h, h)
    key1, value1 = cross._memory_kv(adjusted, h)
    key_delta = (key1 - key0).norm().item()
    value_delta = (value1 - value0).abs().max().item()
    assert key_delta > 0 and value_delta == 0

    altered_padding = h.clone()
    altered_padding[~content] = 1000 * torch.randn_like(altered_padding[~content])
    padding_check = local_difference(altered_padding, content, a_probe, b_probe)
    boundary_error = (padding_check[content] - adjusted[content]).abs().max().item()
    assert boundary_error == 0

    constant = h[:, :1].expand_as(h).clone()
    constant_error = (local_difference(constant, content, a_probe, b_probe) - constant).abs().max().item()
    assert constant_error == 0

    # Holding the center vector fixed, changing a neighbor changes its key.
    neighbor_changed = h.clone()
    neighbor_changed[:, 1] += torch.randn_like(h[:, 1])
    updated = local_difference(neighbor_changed, content, a_probe, b_probe)
    neighbor_key, _ = cross._memory_kv(updated, h)
    center_key_delta = (neighbor_key[:, :, 2] - key1[:, :, 2]).norm().item()
    assert center_key_delta > 0

    # Exact linear absorption in the deliberately normalization-free toy case.
    p = torch.randn(24, 24, dtype=h.dtype)
    wk = torch.randn(24, 12, dtype=h.dtype)
    linear_error = ((h @ p) @ wk - h @ (p @ wk)).abs().max().item()
    assert linear_error < 1e-10

    result = {
        "scope": "random tensors + randomly initialized tiny actual cross-attention; not training efficacy",
        "identity_memory_max_error": initial_error,
        "identity_cross_max_error": initial_cross_error,
        "first_backward_left_gradient_norm": grad_a,
        "first_backward_right_gradient_norm": grad_b,
        "normalized_key_delta_norm": key_delta,
        "anchored_value_max_error": value_delta,
        "masked_neighbor_content_max_error": boundary_error,
        "constant_region_max_error": constant_error,
        "same_center_changed_neighbor_key_delta_norm": center_key_delta,
        "linear_absorption_without_norm_max_error": linear_error,
    }
    Path(__file__).with_name("probe_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
