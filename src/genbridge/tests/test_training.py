import torch
from genbridge.training import build_optimizer


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.decoder = torch.nn.Linear(4, 4)
        self.bridge = torch.nn.Linear(4, 4)
        self.cross_gate = torch.nn.Parameter(torch.zeros(()))


def test_warmup_and_full_finetune_use_one_lr_per_stage():
    model = _TinyModel()
    config = {
        "adapter_warmup_lr": 1.0e-4,
        "full_lr": 8.0e-6,
        "optimizer": "adamw_torch",
        "fused_optimizer": False,
    }
    _, warm_scheduler = build_optimizer(model, config, total_steps=10, stage="interface_warmup")
    assert set(warm_scheduler.base_lrs) == {1.0e-4}
    _, full_scheduler = build_optimizer(model, config, total_steps=10, stage="full_finetune")
    assert set(full_scheduler.base_lrs) == {8.0e-6}
