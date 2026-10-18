"""Offline two-process integration check, launched by test_distributed.py."""

import argparse

import torch
import torch.distributed as dist
from seam_grpo.config import load

from seam_grpo import train as runtime


class ToyReward:
    """Non-tied group rewards exercise the RL gradient in the tiny fixture."""

    def __init__(self, config):
        self.count = 0

    def __call__(self, text, reference):
        value = 0.2 if self.count % 2 else 0.8
        self.count += 1
        return {k: value for k in ("reward", "rouge1", "rouge2", "rougeL")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    settings = load(args.config)
    if settings["training"]["rollout_backend"] == "seam_vllm":
        from fake_vllm import install

        install()
    runtime.RougeReward = ToyReward
    original_step = torch.optim.AdamW.step

    def checked_step(optimizer, *args, **kwargs):
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        grads = torch.cat(
            [p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in parameters]
        )
        peers = [torch.empty_like(grads) for _ in range(dist.get_world_size())]
        dist.all_gather(peers, grads)
        for other in peers:
            torch.testing.assert_close(grads, other, rtol=0, atol=0)
        assert torch.isfinite(grads).all() and grads.abs().sum() > 0
        result = original_step(optimizer, *args, **kwargs)
        weights = torch.cat([p.detach().flatten() for p in parameters])
        dist.all_gather(peers, weights)
        for other in peers:
            torch.testing.assert_close(weights, other, rtol=0, atol=0)
        return result

    torch.optim.AdamW.step = checked_step
    runtime.train(settings, "cpu", overwrite_output_dir=args.overwrite_output_dir)


if __name__ == "__main__":
    main()
