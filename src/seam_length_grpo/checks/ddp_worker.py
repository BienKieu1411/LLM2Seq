"""Real TRL + Gloo integration; no downloaded weights or CUDA kernels."""

import argparse

import torch
import torch.distributed as dist
from fixtures import NumeralTokenizer
from seam_length_grpo.config import load

from seam_length_grpo import train as runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    tokenizer = NumeralTokenizer()
    runtime._tokenizers = lambda config: (tokenizer, tokenizer)
    settings = load(args.config)
    if settings["training"]["rollout_backend"] == "seam_vllm":
        from fake_vllm import install

        install()
    original_step = torch.optim.AdamW.step

    def checked_step(optimizer, *args, **kwargs):
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        grads = torch.cat(
            [p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in parameters]
        )
        peers = [torch.empty_like(grads) for _ in range(dist.get_world_size())]
        dist.all_gather(peers, grads)
        for peer in peers:
            torch.testing.assert_close(grads, peer, rtol=0, atol=0)
        assert torch.isfinite(grads).all() and grads.abs().sum() > 0
        result = original_step(optimizer, *args, **kwargs)
        weights = torch.cat([p.detach().flatten() for p in parameters])
        dist.all_gather(peers, weights)
        for peer in peers:
            torch.testing.assert_close(weights, peer, rtol=0, atol=0)
        return result

    torch.optim.AdamW.step = checked_step
    runtime.train(settings, "cpu")


if __name__ == "__main__":
    main()
