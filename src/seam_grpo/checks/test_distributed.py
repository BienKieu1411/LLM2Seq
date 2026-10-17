import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.training.checkpoint import save_checkpoint
from test_policy import tiny


@pytest.mark.parametrize("ce_weight", [0.0, 0.05])
@pytest.mark.parametrize("rollout_backend", ["native", "seam_vllm"])
def test_two_process_gradient_and_weight_synchronization(tmp_path, ce_weight, rollout_backend):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    cfg, _, _ = tiny()
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"id": str(i), "text": "alpha beta gamma", "summary": "alpha beta"}) + "\n"
                for i in range(3 if split == "train" else 1)
            )
        )
        cfg["data"][f"{split}_file"] = str(path)
    cfg.pop("_meta", None)
    base = tmp_path / "sft.yaml"
    base.write_text(yaml.safe_dump(cfg))
    checkpoint = tmp_path / "sft.pt"
    save_checkpoint(checkpoint, EviSeqAFMR(cfg), None, cfg, epoch=4, step=100)
    grpo = tmp_path / "grpo.yaml"
    grpo.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base),
                "sft_checkpoint": str(checkpoint),
                "output_dir": str(tmp_path / "output"),
                "training": {
                    "rollout_backend": rollout_backend,
                    "max_steps": 1,
                    "group_size": 2,
                    "max_new_tokens": 3,
                    "gradient_accumulation_steps": 2,
                    "validation_batch_size": 1,
                    "ce_weight": ce_weight,
                },
                "reward": {},
            }
        )
    )
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(root), str(root.parent / "eviseq_new")]),
        "OMP_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
            "--nproc_per_node=2",
            str(root / "checks/ddp_worker.py"),
            "--config",
            str(grpo),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads((tmp_path / "output/training_metrics.jsonl").read_text().splitlines()[0])
    assert metrics["step"] == 1 and metrics["grad_norm"] > 0
    assert metrics["behavior_logp_mae"] < 1e-4
