import torch
from eviseq_afmr.modeling.afmr import AdaptiveFullMemoryResidualBridge
from eviseq_afmr.modeling.outputs import EncoderState
from eviseq_afmr.training.optimizer import set_stage_trainability


def _bridge():
    return AdaptiveFullMemoryResidualBridge(
        10,
        10,
        {
            "controller_dim": 6,
            "depth_taps": 0,
            "depth_rank": 4,
            "feature_rank": 4,
            "focus_hidden": 6,
            "focus_windows": [4, 8],
            "focus_strength_max": 1.0,
            "focus_strength_init": 0.1,
            "depth_gate_max": 0.15,
            "depth_gate_init": 0.02,
            "feature_gate_max": 0.2,
            "feature_gate_init": 0.02,
            "temperature_init": 1.0,
            "temperature_min": 0.5,
            "temperature_max": 2.0,
        },
    )


def test_same_bridge_graph_is_deterministic_in_eval():
    bridge = _bridge().eval()
    final = torch.randn(2, 15, 10)
    valid = torch.ones(2, 15, dtype=torch.bool)
    content = valid.clone()
    content[:, :3] = False
    content[:, -1] = False
    state = EncoderState(final, (), valid, content)
    prompt = torch.randn(2, 4, 10)
    prompt_mask = torch.ones(2, 4, dtype=torch.bool)
    budget = torch.tensor([64.0, 128.0])
    first = bridge(state, prompt, prompt_mask, budget)
    second = bridge(state, prompt, prompt_mask, budget)
    assert torch.equal(first.source_bias, second.source_bias)
    assert torch.equal(first.memory, second.memory)


def test_warmup_freezes_pretrained_paths_and_keeps_bridge_trainable():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.bridge = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Module()
            self.decoder.cross = torch.nn.Linear(2, 2)
            self.decoder.base = torch.nn.Linear(2, 2)

    model = Model()
    set_stage_trainability(model, "interface_warmup")
    assert not model.encoder.weight.requires_grad
    assert model.bridge.weight.requires_grad
    assert model.decoder.cross.weight.requires_grad
    assert not model.decoder.base.weight.requires_grad
