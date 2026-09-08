"""Stochastic dropout on two actual processes: synchronization and exact resume."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from eviseq_update.distributed import rank, training_process_group
from eviseq_update.runtime import _write_resolved_config, train
from test_supervised_training import config


def worker(path):
    torch.set_num_threads(1)
    root = Path(path).parent
    with training_process_group("cpu"):
        original_step = torch.optim.AdamW.step
        checked_steps, changed_steps = 0, 0

        def synchronized(values):
            flat = torch.cat([value.detach().float().reshape(-1) for value in values])
            assert torch.isfinite(flat).all()
            reference = flat.clone()
            torch.distributed.broadcast(reference, src=0)
            torch.testing.assert_close(flat, reference, atol=0, rtol=0)

        def recording_step(optimizer, *args, **kwargs):
            nonlocal checked_steps, changed_steps
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            synchronized([torch.tensor([p.grad is not None for p in parameters])])
            synchronized([p.grad for p in parameters if p.grad is not None])
            before = [p.detach().clone() for p in parameters]
            result = original_step(optimizer, *args, **kwargs)
            synchronized(parameters)
            checked_steps += 1
            changed_steps += any(not torch.equal(a, b) for a, b in zip(before, parameters))
            return result

        torch.optim.AdamW.step = recording_step
        train(path, device="cpu")
        train(
            path,
            device="cpu",
            resume_checkpoint=str(root / "fit/epoch_002.pt"),
            output_dir_override=str(root / "resumed"),
        )
        (root / f"rank_{rank()}.json").write_text(json.dumps({"checked": checked_steps, "changed": changed_steps}))


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Requires Gloo")
def test_two_process_stochastic_training_and_resume(tmp_path):
    cfg = config()
    cfg["decoder"]["attention_dropout"] = 0.1
    cfg["experiment"]["output_dir"] = str(tmp_path / "fit")
    cfg["training"].update(
        batch_size=1,
        gradient_accumulation_steps=2,
        full_finetune_epochs=2,
        log_every_steps=1,
        max_grad_norm=1.0,
        save_best=True,
        lr_warmup_ratio=0.25,
    )
    rows = [{"id": str(i), "source": "alpha beta gamma " * (i + 1), "target": "beta " * (i + 1)} for i in range(5)]
    for split, count in (("train", 5), ("validation", 3), ("test", 2)):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows[:count]))
        cfg["data"][f"{split}_file"] = str(path)
    _write_resolved_config(cfg, tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "--nproc_per_node=2",
            str(Path(__file__).resolve()),
            str(tmp_path / "resolved_config.yaml"),
        ],
        env=dict(os.environ, OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    a = torch.load(tmp_path / "fit/last.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    assert a["step"] == b["step"] == 6
    for name, parameter in a["model"].items():
        torch.testing.assert_close(parameter, b["model"][name], atol=0, rtol=0, msg=name)
    rows = [json.loads(line) for line in (tmp_path / "fit/training_metrics.jsonl").read_text().splitlines()]
    steps = [row for row in rows if row["type"] == "step"]
    assert sum(row["examples"] for row in steps) == 15
    assert all(row["world_size"] == 2 for row in steps)
    assert all("neftune_noise_alpha" not in row for row in steps)
    assert a["training_spec"]["decoder_attention_dropout"] == 0.1
    for worker_rank in range(2):
        report = json.loads((tmp_path / f"rank_{worker_rank}.json").read_text())
        assert report["checked"] == 8 and report["changed"] >= 5


if __name__ == "__main__":
    worker(sys.argv[1])
