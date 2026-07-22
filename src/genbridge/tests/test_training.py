import tempfile
from pathlib import Path

import pytest
import torch
from genbridge.training import LengthBucketBatchSampler, build_optimizer, train


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
    warm_optimizer, warm_scheduler = build_optimizer(
        model, config, total_steps=10, stage="interface_warmup"
    )
    assert set(warm_scheduler.base_lrs) == {1.0e-4}
    assert warm_optimizer.defaults["betas"] == (0.9, 0.95)
    _, full_scheduler = build_optimizer(model, config, total_steps=10, stage="full_finetune")
    assert set(full_scheduler.base_lrs) == {8.0e-6}


def test_length_bucket_sampler_covers_data_and_changes_epoch_order():
    lengths = list(range(1, 101))
    sampler = LengthBucketBatchSampler(
        lengths,
        batch_size=10,
        seed=7,
        bucket_multiplier=10,
    )
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert sorted(index for batch in epoch_zero for index in batch) == list(range(100))
    assert epoch_zero != epoch_one
    # With one global mega-bucket, every batch spans only nearby lengths.
    assert max(max(lengths[index] for index in batch) - min(lengths[index] for index in batch) for batch in epoch_zero) <= 9


def test_training_refuses_to_reuse_a_final_checkpoint_directory():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "run"
        output.mkdir()
        (output / "final.pt").touch()
        config = root / "config.yaml"
        config.write_text(
            "experiment:\n"
            f"  output_dir: {output}\n"
            "training:\n"
            "  seed: 1\n",
            encoding="utf-8",
        )
        with pytest.raises(FileExistsError, match="Refusing to mix a new run"):
            train(str(config))
