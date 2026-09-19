import torch
from eviseq_afmr.modeling.afmr import AdaptiveFullMemoryResidualBridge
from eviseq_afmr.modeling.outputs import EncoderState


def _bridge() -> AdaptiveFullMemoryResidualBridge:
    return AdaptiveFullMemoryResidualBridge(
        16,
        16,
        {
            "controller_dim": 8,
            "depth_taps": 2,
            "depth_rank": 4,
            "feature_rank": 4,
            "focus_hidden": 8,
            "focus_windows": [4, 8],
            "focus_overlap": 0.5,
            "depth_gate_init": 0.02,
            "depth_gate_max": 0.15,
            "feature_gate_init": 0.02,
            "feature_gate_max": 0.20,
            "focus_strength_init": 0.10,
            "focus_strength_max": 1.0,
            "temperature_init": 1.0,
            "temperature_min": 0.5,
            "temperature_max": 2.0,
        },
    )


def test_afmr_shapes_and_special_token_masking():
    bridge = _bridge()
    final = torch.randn(2, 13, 16)
    valid = torch.ones(2, 13, dtype=torch.bool)
    content = valid.clone()
    content[:, :2] = False
    content[:, -1] = False
    state = EncoderState(final, (final + 0.1, final), valid, content)
    output = bridge(state, torch.randn(2, 3, 16), torch.ones(2, 3, dtype=torch.bool), torch.tensor([32.0, 64.0]))
    assert output.memory.shape == final.shape
    assert output.source_bias.shape == valid.shape
    assert torch.equal(output.memory_mask, valid)
    assert torch.equal(output.content_mask, content)
    assert torch.all(output.source_bias[:, :2] == 0)
    assert torch.all(output.source_bias[:, -1] == 0)
    assert torch.isfinite(output.source_bias).all()


def test_zero_initialized_residual_preserves_pretrained_memory():
    bridge = _bridge()
    final = torch.randn(1, 11, 16)
    valid = torch.ones(1, 11, dtype=torch.bool)
    content = valid.clone()
    content[:, 0] = False
    content[:, -1] = False
    output = bridge(
        EncoderState(final, (final + 1, final), valid, content),
        torch.randn(1, 2, 16),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([64.0]),
    )
    assert torch.allclose(output.memory, final, atol=1.0e-6)


def test_focus_prior_is_content_normalized_and_ignores_prefix_values():
    torch.manual_seed(3)
    bridge = _bridge()
    with torch.no_grad():
        bridge.focus_output.weight.normal_()
    final = torch.randn(1, 17, 16)
    valid = torch.ones(1, 17, dtype=torch.bool)
    content = valid.clone()
    content[:, :3] = False
    content[:, -1] = False
    prompt = torch.randn(1, 3, 16)
    prompt_mask = torch.ones(1, 3, dtype=torch.bool)
    state = EncoderState(final, (final + 0.1, final), valid, content)
    first = bridge(state, prompt, prompt_mask, torch.tensor([32.0]))
    changed = final.clone()
    changed[:, :3] += 1000.0
    second = bridge(
        EncoderState(changed, (changed + 0.1, changed), valid, content), prompt, prompt_mask, torch.tensor([32.0])
    )
    assert torch.allclose(first.source_bias[:, 3:-1], second.source_bias[:, 3:-1], atol=1.0e-5)
    assert torch.all(first.source_bias[:, :3] == 0)
    assert torch.allclose(first.source_bias[:, -1], torch.zeros(1))
    exp_mean = first.source_bias[:, 3:-1].float().exp().mean()
    assert torch.allclose(exp_mean, torch.ones_like(exp_mean), atol=1.0e-5)


def test_cross_space_projection_accepts_bfloat16_encoder_states():
    bridge = AdaptiveFullMemoryResidualBridge(
        8,
        10,
        {
            "controller_dim": 8,
            "depth_taps": 1,
            "depth_rank": 4,
            "feature_rank": 4,
            "focus_hidden": 8,
            "focus_windows": [4, 8],
            "focus_strength_init": 0.10,
            "focus_strength_max": 1.0,
            "depth_gate_init": 0.02,
            "depth_gate_max": 0.15,
            "feature_gate_init": 0.02,
            "feature_gate_max": 0.20,
            "temperature_init": 1.0,
            "temperature_min": 0.5,
            "temperature_max": 2.0,
        },
    )
    valid = torch.ones(1, 12, dtype=torch.bool)
    content = valid.clone()
    content[:, 0] = False
    output = bridge(
        EncoderState(
            torch.randn(1, 12, 8, dtype=torch.bfloat16), (torch.randn(1, 12, 8, dtype=torch.bfloat16),), valid, content
        ),
        torch.randn(1, 2, 10),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([32.0]),
    )
    assert output.memory.shape == (1, 12, 10)
    assert torch.isfinite(output.memory).all()
    assert bridge.feature_gate_raw.bias.shape == (10,)
