"""Private single-client rollout host on a GPU separate from the trainer."""

from __future__ import annotations

import argparse
import os
import traceback
from multiprocessing.connection import Listener
from types import SimpleNamespace

import torch
from eviseq_afmr.training.checkpoint import architecture_spec

from .remote_rollout import pack_outputs, transfer_communicator


class RolloutSession:
    def __init__(self, connection):
        self.connection = connection
        self.engine = self.comm = None
        self.weights = self.manifest = None
        self.version = -1

    def initialize(self, message):
        from seam_vllm import SEAMLLM

        if self.engine is not None:
            raise RuntimeError("Rollout host is already initialized")
        torch.cuda.set_device(0)
        uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
        if uuid and uuid == message["train_gpu_uuid"]:
            raise ValueError("Training and rollout host must use different physical GPUs")
        self.engine = SEAMLLM(message["config_path"], message["checkpoint"], **message["engine_options"])
        torch.backends.cuda.matmul.allow_tf32 = bool(self.engine.config["training"].get("tf32", False))
        return {"gpu_uuid": uuid}

    def sync(self, message):
        if self.comm is None or message["version"] <= self.version:
            raise RuntimeError("Initialize NCCL and use a strictly newer actor version")
        if architecture_spec(self.engine.config) != message["spec"]:
            raise ValueError("Actor architecture differs from dedicated rollout host")
        manifest = message["manifest"]
        if self.manifest is None:
            self.weights = [
                (name, torch.empty(shape, dtype=getattr(torch, dtype), device="cuda:0"))
                for name, shape, dtype in manifest
            ]
            self.manifest = manifest
        elif manifest != self.manifest:
            raise ValueError("Actor parameter names/shapes/dtypes changed during training")
        self.connection.send({"result": "ready_for_weights"})
        for _, value in self.weights:
            self.comm.broadcast(value, src=0)
        torch.cuda.current_stream().synchronize()
        actor = SimpleNamespace(config=self.engine.config, named_parameters=lambda: iter(self.weights))
        result = self.engine.sync_weights(actor, policy_version=message["version"])
        self.version = result["policy_version"]
        return result

    def dispatch(self, message):
        operation = message["op"]
        if operation == "initialize":
            return self.initialize(message)
        if operation == "init_transfer":
            self.comm = transfer_communicator(message["port"], 1, torch.device("cuda:0"), message["timeout"])
            return "transfer_ready"
        if operation == "sync":
            return self.sync(message)
        if operation == "generate":
            if self.version < 0 or message["version"] != self.version:
                raise RuntimeError("Generation requested a stale or unsynchronized actor")
            generate = self.engine.generate if message["mode"] == "source" else self.engine.engine.generate
            return pack_outputs(generate(message["prompts"], message["params"], **message["options"]))
        raise ValueError(f"Unknown rollout operation: {operation}")

    def close(self):
        if self.engine is not None:
            self.engine.engine.llm_engine.engine_core.shutdown()
            self.engine = None
        if self.comm is not None:
            self.comm.destroy()
            self.comm = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    auth = bytes.fromhex(os.environ.pop("SEAM_ROLLOUT_AUTH"))
    # Loopback + per-launch authentication: no arbitrary remote control or
    # weight loading endpoint is exposed by the public inference HTTP server.
    with Listener(("127.0.0.1", args.port), authkey=auth) as listener:
        with listener.accept() as connection:
            session = RolloutSession(connection)
            try:
                while True:
                    message = connection.recv()
                    if message["op"] == "shutdown":
                        session.close()
                        connection.send({"result": "stopped"})
                        break
                    connection.send({"result": session.dispatch(message)})
            except EOFError:
                pass
            except Exception as exc:
                traceback.print_exc()
                connection.send({"error": f"{type(exc).__name__}: {exc}"})
            finally:
                session.close()


if __name__ == "__main__":
    main()
