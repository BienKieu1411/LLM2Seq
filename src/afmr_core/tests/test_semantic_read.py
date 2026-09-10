import torch

from afmr_core.modeling.semantic_read import SemanticReader, smooth_relative_rms_cap


def _state(key_source="H0", empty=False):
    torch.manual_seed(3)
    reader = SemanticReader(6, rank=4, key_source=key_source, value_source="H0", output_init="tiny_rms_1e-3")
    h0 = torch.randn(2, 5, 6)
    memory = h0 + 2.0 * torch.randn_like(h0)
    mask = torch.ones(2, 5, dtype=torch.bool)
    if empty:
        mask[1].fill_(False)
    bias = torch.zeros(2, 5)
    return reader, h0, memory, mask, bias


def test_flat_reader_shapes_and_relative_rms_bound():
    reader, h0, memory, mask, bias = _state()
    state = reader.prepare(H0=h0, M=memory, source_mask=mask, source_bias=bias)
    hidden = torch.randn(2, 3, 6, requires_grad=True)
    hs, diagnostics = reader.read(hidden, state)
    assert hs.shape == hidden.shape
    assert diagnostics["q"].shape == (2, 3, 4)
    assert diagnostics["u"].shape == (2, 3, 4)
    ratio = torch.sqrt(
        diagnostics["delta"].float().square().mean(-1) / hidden.float().square().mean(-1).clamp_min(1e-12)
    )
    assert bool((ratio <= 0.10001).all())
    hs.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_empty_source_is_finite_and_has_zero_residual():
    reader, h0, memory, mask, bias = _state(empty=True)
    state = reader.prepare(H0=h0, M=memory, source_mask=mask, source_bias=bias)
    hidden = torch.randn(2, 2, 6)
    hs, diagnostics = reader.read(hidden, state)
    assert torch.isfinite(hs).all()
    assert torch.allclose(hs[1], hidden[1])
    assert torch.allclose(diagnostics["evidence"][1], torch.zeros_like(diagnostics["evidence"][1]))
    assert torch.allclose(diagnostics["attention"][1], torch.zeros_like(diagnostics["attention"][1]))


def test_key_source_variant_changes_keys_but_values_stay_h0():
    reader_h0, h0, memory, mask, bias = _state("H0")
    reader_m, _, _, _, _ = _state("M")
    state_h0 = reader_h0.prepare(H0=h0, M=memory, source_mask=mask, source_bias=bias)
    state_m = reader_m.prepare(H0=h0, M=memory, source_mask=mask, source_bias=bias)
    assert torch.allclose(state_h0.value_memory, state_m.value_memory)
    assert not torch.allclose(state_h0.key_memory, state_m.key_memory)


def test_bfloat16_reader_matches_projection_dtypes():
    reader, h0, memory, mask, bias = _state()
    reader = reader.to(dtype=torch.bfloat16)
    state = reader.prepare(
        H0=h0.to(dtype=torch.bfloat16),
        M=memory.to(dtype=torch.bfloat16),
        source_mask=mask,
        source_bias=bias,
    )
    hidden = torch.randn(2, 3, 6, dtype=torch.bfloat16, requires_grad=True)
    fused, diagnostics = reader.read(hidden, state)
    assert fused.dtype == torch.bfloat16
    assert torch.isfinite(fused.float()).all()
    fused.float().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad.float()).all()


def test_semantic_state_reorder_is_exact():
    reader, h0, memory, mask, bias = _state()
    state = reader.prepare(H0=h0, M=memory, source_mask=mask, source_bias=bias)
    indices = torch.tensor([1, 0])
    reordered = state.index_select(indices)
    assert torch.equal(reordered.source_mask, state.source_mask.index_select(0, indices))
    hidden = torch.randn(2, 2, 6)
    original, _ = reader.read(hidden, reordered)
    expected, _ = reader.read(hidden, state)
    assert original.shape == expected.shape


def test_smooth_cap_zero_and_large_residual_are_safe():
    hidden = torch.randn(2, 3, 6, requires_grad=True)
    zero = torch.zeros_like(hidden)
    assert torch.equal(smooth_relative_rms_cap(zero, hidden), zero)
    delta = 10000 * torch.randn_like(hidden)
    bounded = smooth_relative_rms_cap(delta, hidden)
    ratio = torch.sqrt(bounded.float().square().mean(-1) / hidden.float().square().mean(-1).clamp_min(1e-12))
    assert bool((ratio <= 0.100001).all())
    bounded.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
