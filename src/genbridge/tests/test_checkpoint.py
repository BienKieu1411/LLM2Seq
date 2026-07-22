import tempfile
from pathlib import Path

import pytest
import torch
from genbridge.checkpoint import load_checkpoint, save_checkpoint
from genbridge.evaluate import evaluate


class _CheckpointModel(torch.nn.Module):
    def __init__(self, with_extra: bool = False):
        super().__init__()
        self.core = torch.nn.Linear(3, 3)
        if with_extra:
            self.extra = torch.nn.Parameter(torch.ones(3))


def test_full_checkpoint_rejects_silent_architecture_drift():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "full.pt"
        save_checkpoint(_CheckpointModel(), path, {}, epoch=1, global_step=2)
        with pytest.raises(RuntimeError, match="missing parameter tensors: extra"):
            load_checkpoint(_CheckpointModel(with_extra=True), path)


def test_full_checkpoint_round_trip():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "full.pt"
        source = _CheckpointModel()
        save_checkpoint(source, path, {}, epoch=3, global_step=9)
        target = _CheckpointModel()
        payload = load_checkpoint(target, path)
        assert payload["stores_full_parameter_state"] is True
        assert payload["epoch"] == 3
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            torch.testing.assert_close(source_parameter, target_parameter)


def test_evaluation_rejects_checkpoint_from_running_job():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "final.pt"
        checkpoint.touch()
        (root / "RUNNING").write_text("running\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="indicates an incomplete run"):
            evaluate("missing.yaml", str(checkpoint), str(root / "out.jsonl"), None)
