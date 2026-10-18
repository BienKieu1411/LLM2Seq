"""CPU transport/transaction checks; NCCL and real vLLM still need two CUDA GPUs."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.training.checkpoint import architecture_spec
from seam_grpo.remote_rollout import RemoteSEAMLLM, pack_outputs, unpack_outputs
from seam_grpo.rollout_worker import RolloutSession
from seam_grpo.train import check_rollout_device
from seam_grpo.vllm_rollout import completion_rows
from test_policy import tiny


def test_receiver_transfers_complete_actor_and_rejects_stale_generation(monkeypatch):
    cfg, _, _ = tiny()
    actor = EviSeqAFMR(cfg).eval()
    weights = list(actor.named_parameters())
    manifest = [(name, list(value.shape), str(value.dtype).removeprefix("torch.")) for name, value in weights]
    messages, restored = [], {}
    session = RolloutSession(SimpleNamespace(send=messages.append))
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: original_empty(*a, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: SimpleNamespace(synchronize=lambda: None))
    received = []

    def broadcast(destination, src):
        assert src == 0
        name, reference = weights[len(received)]
        destination.copy_(reference.detach())
        received.append(name)

    def sync(proxy, policy_version):
        restored.update({name: value.clone() for name, value in proxy.named_parameters()})
        return {"policy_version": policy_version, "tensors": len(restored)}

    session.comm = SimpleNamespace(broadcast=broadcast)
    session.engine = SimpleNamespace(config=cfg, sync_weights=sync)
    message = {"op": "sync", "manifest": manifest, "spec": architecture_spec(cfg), "version": 0}
    assert session.dispatch(message)["policy_version"] == 0
    assert messages == [{"result": "ready_for_weights"}]
    assert received == [name for name, _ in weights]
    for name, value in weights:
        torch.testing.assert_close(restored[name], value, atol=0, rtol=0)
    assert all(any(name.startswith(prefix) for name in restored) for prefix in ("encoder.", "bridge.", "decoder."))
    with pytest.raises(RuntimeError, match="newer"):
        session.dispatch(message)
    with pytest.raises(RuntimeError, match="stale"):
        session.dispatch({"op": "generate", "version": 1})
    with pytest.raises(ValueError, match="architecture"):
        session.dispatch({**message, "version": 1, "spec": {}})


def test_sender_requires_acknowledged_complete_version(monkeypatch):
    cfg, _, _ = tiny()
    actor = EviSeqAFMR(cfg)
    client = object.__new__(RemoteSEAMLLM)
    client.device, client.version, client.failed = torch.device("cpu"), -1, False
    sent, broadcasts = [], []
    client._request = lambda operation, **kw: sent.append((operation, kw))
    client._receive = lambda: {"policy_version": 0}
    client.comm = SimpleNamespace(broadcast=lambda value, src: broadcasts.append(value.clone()))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: SimpleNamespace(synchronize=lambda: None))
    client.sync_weights(actor, policy_version=0)
    assert client.version == 0 and not client.failed
    assert len(sent[0][1]["manifest"]) == len(broadcasts) == len(list(actor.named_parameters()))
    for actual, (_, expected) in zip(broadcasts, actor.named_parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(RuntimeError, match="stale"):
        client.sync_weights(actor, policy_version=0)
    client._receive = lambda: {"policy_version": 7}
    with pytest.raises(RuntimeError, match="wrong actor version"):
        client.sync_weights(actor, policy_version=1)
    assert client.failed
    with pytest.raises(RuntimeError, match="Synchronize"):
        client.generate(["source"], None)


def test_dedicated_output_roundtrip_retains_group_order_eos_and_actual_logps():
    def candidate(token):
        return SimpleNamespace(
            token_ids=[token, 2],
            logprobs=[{token: SimpleNamespace(logprob=-0.4)}, {2: SimpleNamespace(logprob=-0.8)}],
        )

    outputs = [SimpleNamespace(prompt_token_ids=[9], outputs=[candidate(4), candidate(5)])]
    actual = completion_rows(unpack_outputs(pack_outputs(outputs)), [[9], [9]], eos_token_id=2, budget=3, group_size=2)
    assert actual["completion_ids"] == [[4, 2], [5, 2]]
    assert actual["logprobs"] == [[-0.4, -0.8], [-0.4, -0.8]]


def test_dedicated_host_is_not_a_second_training_rank(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one visible training GPU"):
        check_rollout_device({"training": {"rollout_backend": "seam_vllm", "vllm_mode": "dedicated"}}, None)


def test_two_gpu_launcher_selects_single_trainer_and_dedicated_host(tmp_path):
    python = tmp_path / "python"
    python.write_text('#!/bin/bash\necho "VISIBLE=$CUDA_VISIBLE_DEVICES NPROC=$NPROC_PER_NODE ARGS=$*"\n')
    python.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts/run_grpo_two_gpu.sh"
    result = subprocess.run(
        ["bash", str(script), "--config", "example.yaml"],
        env={**os.environ, "PYTHON_BIN": str(python), "TRAIN_GPU": "2", "ROLLOUT_GPU": "3", "NPROC_PER_NODE": "2"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "VISIBLE=2 NPROC=1" in result.stdout
    assert "--vllm-mode dedicated --rollout-gpu 3" in result.stdout
    assert "torch.distributed.run" not in result.stdout


def test_real_two_process_protocol_transfers_versions_and_stops_child(tmp_path, monkeypatch):
    import sys

    import yaml
    from remote_cpu_worker import CPUCommunicator

    from seam_grpo import remote_rollout

    if not torch.distributed.is_gloo_available():
        pytest.skip("CPU Gloo unavailable")
    cfg, _, _ = tiny()
    actor = EviSeqAFMR(cfg)
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k != "_meta"}))
    checkpoint = tmp_path / "sft.pt"
    torch.save({"model": actor.state_dict()}, checkpoint)
    helper = Path(__file__).with_name("remote_cpu_worker.py")
    original_popen = subprocess.Popen

    def launch_test_worker(command, **kwargs):
        return original_popen([sys.executable, str(helper), "--port", command[-1]], **kwargs)

    monkeypatch.setattr(remote_rollout.subprocess, "Popen", launch_test_worker)
    monkeypatch.setattr(remote_rollout, "transfer_communicator", CPUCommunicator)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(uuid="train-gpu"))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: SimpleNamespace(synchronize=lambda: None))
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(num_processes=1, device=torch.device("cpu")),
        settings={"vllm_server_timeout": 30, "vllm_rollout_gpu": "1"},
        args=SimpleNamespace(output_dir=str(tmp_path)),
    )
    client = RemoteSEAMLLM(config_path, checkpoint, trainer=trainer)
    process = client.process
    try:
        with torch.no_grad():
            next(actor.parameters()).fill_(0)
        client.sync_weights(actor, policy_version=0)
        outputs = client.generate(["source"], SimpleNamespace(n=2))
        assert len(outputs[0].outputs) == 2
        assert outputs[0].outputs[0].logprobs[0][4].logprob == pytest.approx(-0.1)
        with torch.no_grad():
            next(actor.parameters()).fill_(1)
        client.sync_weights(actor, policy_version=1)
        outputs = client.generate(["source"], SimpleNamespace(n=1))
        assert outputs[0].outputs[0].logprobs[0][4].logprob == pytest.approx(-1.1)
        assert not torch.distributed.is_initialized()  # inference peer never becomes a trainer rank
    finally:
        client.close()
    assert process.poll() == 0
    assert client.comm is client.connection is client.process is None
