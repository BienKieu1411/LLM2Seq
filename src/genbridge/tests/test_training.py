import tempfile
from pathlib import Path

import pytest
import torch
from genbridge.training import (
    LengthBucketBatchSampler,
    build_optimizer,
    evaluate_teacher_forced,
    train,
)
from torch.utils.data import DataLoader


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


def test_training_refuses_to_reuse_a_checkpoint_directory():
    for checkpoint_name in ("best.pt", "last.pt", "final.pt"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            output.mkdir()
            (output / checkpoint_name).touch()
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


class _ValidationModel(torch.nn.Module):
    def forward(self, labels, **kwargs):
        del kwargs
        supervised = labels.ne(-100)
        # Make the scalar depend on a parameter so this resembles a model
        # output while retaining an exactly checkable token-weighted mean.
        value = labels[supervised].float().mean() + self.anchor * 0.0
        return {"loss": value, "loss_ce": value}

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


def test_validation_ce_is_weighted_by_supervised_target_tokens():
    batches = [
        {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "unit_ids": torch.ones(1, 2, dtype=torch.long),
            "evidence_labels": torch.ones(1, 1),
            "decoder_input_ids": torch.ones(1, 3, dtype=torch.long),
            "decoder_attention_mask": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.tensor([[1, -100, -100]]),
        },
        {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "unit_ids": torch.ones(1, 2, dtype=torch.long),
            "evidence_labels": torch.ones(1, 1),
            "decoder_input_ids": torch.ones(1, 3, dtype=torch.long),
            "decoder_attention_mask": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.tensor([[3, 3, 3]]),
        },
    ]
    metrics = evaluate_teacher_forced(
        _ValidationModel(),
        DataLoader(batches, batch_size=None),
        torch.device("cpu"),
        "encoder_decoder",
        torch.float32,
        False,
    )
    assert metrics["eval_loss_ce"] == pytest.approx(2.5)
    assert metrics["eval_supervised_tokens"] == 4
