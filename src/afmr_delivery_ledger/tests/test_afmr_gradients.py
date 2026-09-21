import torch
from eviseq_afmr.modeling.afmr import AdaptiveFullMemoryResidualBridge
from eviseq_afmr.modeling.outputs import EncoderState


def test_residual_and_focus_paths_receive_gradient():
    bridge = AdaptiveFullMemoryResidualBridge(
        12,
        12,
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
    final = torch.randn(2, 12, 12)
    valid = torch.ones(2, 12, dtype=torch.bool)
    content = valid.clone()
    content[:, :2] = False
    content[:, -1] = False
    prompt = torch.randn(2, 3, 12)
    prompt_mask = torch.ones(2, 3, dtype=torch.bool)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1.0e-2)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = bridge(
            EncoderState(final, (final + 0.2, final), valid, content),
            prompt,
            prompt_mask,
            torch.tensor([32.0, 64.0]),
        )
        loss = (
            output.memory.square().mean()
            + (output.source_bias * torch.arange(12, dtype=output.source_bias.dtype)).mean()
        )
        loss.backward()
        assert bridge.depth_out.weight.grad is not None and bridge.depth_out.weight.grad.abs().sum() > 0
        assert bridge.feature_up.weight.grad is not None and bridge.feature_up.weight.grad.abs().sum() > 0
        if step == 0:
            assert bridge.focus_output.weight.grad is not None and bridge.focus_output.weight.grad.abs().sum() > 0
        optimizer.step()
    assert bridge.focus_query.weight.grad is not None and bridge.focus_query.weight.grad.abs().sum() > 0
    assert bridge.focus_key.weight.grad is not None and bridge.focus_key.weight.grad.abs().sum() > 0
