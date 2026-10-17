import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from fixtures import settings_file


@pytest.mark.parametrize("ce_weight,backend", [(0.0, "native"), (0.05, "native"), (0.05, "seam_vllm")])
def test_two_process_trl_gradients_and_weights(tmp_path, ce_weight, backend):
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    _, _, path = settings_file(tmp_path, source_weight=0.05, backend=backend, max_steps=2)
    raw = yaml.safe_load(path.read_text())
    raw["training"]["ce_weight"] = ce_weight
    path.write_text(yaml.safe_dump(raw))
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(root), str(root.parent / "seam_grpo"), str(root.parent / "eviseq_new")]),
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
            str(path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metrics = [json.loads(line) for line in (tmp_path / "out/training_metrics.jsonl").read_text().splitlines()]
    assert any(row.get("grad_norm", 0) > 0 for row in metrics)
    assert any(row.get("behavior_logp_mae", 1) < 1e-4 for row in metrics)
    assert (tmp_path / "out/last.pt").is_file()


def test_two_process_eval_merges_uneven_shards_in_input_order(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    settings, _, config = settings_file(tmp_path, source_weight=0)
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(line) for line in (tmp_path / "test.jsonl").read_text().splitlines()]
    rows.append({**rows[0], "id": "c"})
    (tmp_path / "test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(root), str(root.parent / "seam_grpo"), str(root.parent / "eviseq_new")]),
        "OMP_NUM_THREADS": "1",
    }
    arguments = [
        str(root / "checks/eval_worker.py"),
        "--config",
        str(config),
        "--checkpoint",
        settings["sft_checkpoint"],
        "--oracle-reference-length",
        "--batch-size",
        "1",
        "--device",
        "cpu",
    ]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    distributed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
            *arguments,
            "--output",
            str(tmp_path / "parallel.jsonl"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert distributed.returncode == 0, distributed.stdout + distributed.stderr
    single = subprocess.run(
        [sys.executable, *arguments, "--output", str(tmp_path / "single.jsonl")],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert single.returncode == 0, single.stdout + single.stderr
    parallel_rows = [json.loads(line) for line in (tmp_path / "parallel.jsonl").read_text().splitlines()]
    assert [row["id"] for row in parallel_rows] == ["a", "b", "c"]
    assert (tmp_path / "parallel.jsonl").read_text() == (tmp_path / "single.jsonl").read_text()
