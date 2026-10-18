"""Two-process CPU substitute for checking the REAL dedicated-host protocol."""

from datetime import timedelta
from types import SimpleNamespace

import torch
import yaml


class CPUCommunicator:
    def __init__(self, port, rank, device, timeout):
        store = torch.distributed.TCPStore(
            "127.0.0.1", port, 2, rank == 0, timedelta(seconds=timeout), wait_for_workers=True
        )
        self.group = torch.distributed.ProcessGroupGloo(store, rank, 2, timedelta(seconds=timeout))

    def broadcast(self, tensor, src):
        options = torch.distributed.BroadcastOptions()
        options.rootRank = src
        self.group.broadcast([tensor], options).wait()

    def destroy(self):
        self.group = None


class FakeLLM:
    def __init__(self, config_path, checkpoint, **kwargs):
        self.config = yaml.safe_load(open(config_path))
        self.value = None
        self.engine = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: None)))

    def sync_weights(self, actor, *, policy_version):
        weights = list(actor.named_parameters())
        assert any(name.startswith("encoder.") for name, _ in weights)
        assert any(name.startswith("bridge.") for name, _ in weights)
        assert any(name.startswith("decoder.grounded_copy.") for name, _ in weights)
        self.value = float(weights[0][1].flatten()[0])
        return {"policy_version": policy_version, "tensors": len(weights)}

    def generate(self, sources, params, **kwargs):
        return [
            SimpleNamespace(
                prompt_token_ids=[1],
                outputs=[
                    SimpleNamespace(
                        token_ids=[4, 2],
                        logprobs=[
                            {4: SimpleNamespace(logprob=-abs(self.value) - 0.1)},
                            {2: SimpleNamespace(logprob=-0.2)},
                        ],
                    )
                    for _ in range(params.n)
                ],
            )
            for _ in sources
        ]


if __name__ == "__main__":
    import seam_vllm

    from seam_grpo import rollout_worker

    seam_vllm.SEAMLLM = FakeLLM
    rollout_worker.transfer_communicator = CPUCommunicator
    torch.cuda.set_device = lambda *a: None
    torch.cuda.get_device_properties = lambda *a: SimpleNamespace(uuid="rollout-gpu")
    torch.cuda.current_stream = lambda *a: SimpleNamespace(synchronize=lambda: None)
    original_empty = torch.empty
    torch.empty = lambda *a, **kw: original_empty(*a, **{**kw, "device": "cpu"})
    rollout_worker.main()
