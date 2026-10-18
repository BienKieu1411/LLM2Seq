"""Dedicated local vLLM process; control over authenticated IPC, weights over NCCL."""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path
from types import SimpleNamespace

import torch


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def transfer_communicator(port, rank, device, timeout):
    # Separate from the trainer's default process group: never turn its single
    # training GPU into a two-rank DDP job with the inference worker.
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    group = StatelessProcessGroup.create("127.0.0.1", port, rank, 2, store_timeout=timeout)
    comm = PyNcclCommunicator(group, device=device)
    if not comm.available or comm.disabled:
        raise RuntimeError("Dedicated SEAM rollout requires an active NCCL transfer communicator")
    return comm


def pack_outputs(outputs):
    """Transfer only action IDs/log-probs, without CUDA tensors or source copies."""
    return [
        {
            "prompt_token_ids": list(output.prompt_token_ids),
            "outputs": [
                {
                    "token_ids": list(completion.token_ids),
                    "logprobs": None
                    if completion.logprobs is None
                    else [
                        {int(token): float(value.logprob) for token, value in row.items()}
                        for row in completion.logprobs
                    ],
                }
                for completion in output.outputs
            ],
        }
        for output in outputs
    ]


def unpack_outputs(outputs):
    return [
        SimpleNamespace(
            prompt_token_ids=output["prompt_token_ids"],
            outputs=[
                SimpleNamespace(
                    token_ids=completion["token_ids"],
                    logprobs=None
                    if completion["logprobs"] is None
                    else [
                        {token: SimpleNamespace(logprob=value) for token, value in row.items()}
                        for row in completion["logprobs"]
                    ],
                )
                for completion in output["outputs"]
            ],
        )
        for output in outputs
    ]


class RemoteSEAMLLM:
    """One trainer GPU and one independently visible inference GPU on this host."""

    def __init__(self, config_path, checkpoint, *, trainer, **engine_options):
        if trainer.accelerator.num_processes != 1 or torch.cuda.device_count() != 1:
            raise ValueError("Dedicated rollout requires one visible TRAIN GPU and NPROC_PER_NODE=1")
        cfg = trainer.settings
        self.timeout = cfg["vllm_server_timeout"]
        self.device = trainer.accelerator.device
        self.version = -1
        self.failed = False
        self.connection = self.process = self.comm = None
        self.log_path = Path(trainer.args.output_dir) / "rollout_server.log"
        auth = secrets.token_bytes(32)
        port = free_port()
        root = Path(__file__).resolve().parents[1]
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": cfg["vllm_rollout_gpu"],
            "SEAM_ROLLOUT_AUTH": auth.hex(),
            "PYTHONPATH": os.pathsep.join(
                [str(root), str(root.parent / "eviseq_new"), os.environ.get("PYTHONPATH", "")]
            ),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(free_port()),
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "PYTHONUNBUFFERED": "1",
        }
        try:
            with self.log_path.open("w", encoding="utf-8") as stream:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "seam_grpo.rollout_worker", "--port", str(port)],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            deadline = time.monotonic() + self.timeout
            while self.connection is None:
                if self.process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"SEAM rollout host did not start; inspect {self.log_path}")
                try:
                    self.connection = Client(("127.0.0.1", port), authkey=auth)
                except ConnectionRefusedError:
                    time.sleep(0.1)
            properties = torch.cuda.get_device_properties(self.device)
            self._request(
                "initialize",
                config_path=str(config_path),
                checkpoint=str(checkpoint),
                train_gpu_uuid=str(getattr(properties, "uuid", "")),
                engine_options=engine_options,
            )
            port = free_port()
            self.connection.send({"op": "init_transfer", "port": port, "timeout": self.timeout})
            self.comm = transfer_communicator(port, 0, self.device, self.timeout)
            self._receive()
        except BaseException:
            self.failed = True
            self.close()
            raise
        # Length-conditioned rollout also uses the native engine's raw prompt API.
        self.engine = SimpleNamespace(generate=self.generate_requests)

    def _receive(self):
        try:
            if not self.connection.poll(self.timeout):
                self.failed = True
                raise TimeoutError(f"SEAM rollout host timed out; inspect {self.log_path}")
            response = self.connection.recv()
        except (EOFError, OSError) as exc:
            self.failed = True
            raise RuntimeError(f"SEAM rollout host disconnected; inspect {self.log_path}") from exc
        if "error" in response:
            self.failed = True
            raise RuntimeError(f"SEAM rollout host: {response['error']}; inspect {self.log_path}")
        return response["result"]

    def _request(self, operation, **payload):
        self.connection.send({"op": operation, **payload})
        return self._receive()

    @torch.no_grad()
    def sync_weights(self, model, *, policy_version, **kwargs):
        from eviseq_afmr.training.checkpoint import architecture_spec

        if self.failed or policy_version <= self.version:
            raise RuntimeError("Dedicated SEAM actor sync failed previously or policy version is stale")
        weights = list(model.named_parameters())
        if any(value.device != self.device for _, value in weights):
            raise ValueError("Actor parameters must all be on the training GPU")
        self.failed = True
        self._request(
            "sync",
            version=policy_version,
            spec=architecture_spec(model.config),
            manifest=[(name, list(value.shape), str(value.dtype).removeprefix("torch.")) for name, value in weights],
        )
        # Receiver confirms all buffers/shapes before either peer starts NCCL.
        # Original FP32 values travel GPU-to-GPU; the existing transaction does
        # the validated packed-QKV/tied-vocabulary mapping and inference cast.
        for _, value in weights:
            self.comm.broadcast(value.detach().contiguous(), src=0)
        result = self._receive()
        torch.cuda.current_stream(self.device).synchronize()
        if result["policy_version"] != policy_version:
            raise RuntimeError("Dedicated SEAM host acknowledged the wrong actor version")
        self.version = policy_version
        self.failed = False
        return result

    def _generate(self, mode, prompts, params, options):
        if self.failed or self.version < 0:
            raise RuntimeError("Synchronize the complete actor before dedicated SEAM generation")
        return unpack_outputs(
            self._request("generate", mode=mode, prompts=prompts, params=params, options=options, version=self.version)
        )

    def generate(self, sources, sampling_params, **options):
        return self._generate("source", sources, sampling_params, options)

    def generate_requests(self, requests, sampling_params, **options):
        return self._generate("requests", requests, sampling_params, options)

    def close(self):
        if self.connection is not None:
            try:
                if not self.failed:
                    self.connection.send({"op": "shutdown"})
                    if self.connection.poll(5):
                        self.connection.recv()
            except (EOFError, OSError):
                pass
            self.connection.close()
            self.connection = None
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self.comm is not None:
            self.comm.destroy()
            self.comm = None
