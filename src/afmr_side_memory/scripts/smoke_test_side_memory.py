#!/usr/bin/env python3
"""Offline smoke: tiny local models only, with no checkpoint download."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afmr_side_memory.config import load_config  # noqa: E402
from afmr_side_memory.modeling.model import SideMemorySummarizer  # noqa: E402


def main() -> None:
    config = load_config(ROOT / "configs" / "afmr_smoke.yaml")
    config["model"]["gradient_checkpointing"] = False
    model = SideMemorySummarizer(config)
    source = torch.tensor([[1, 7, 11, 2]])
    target = torch.tensor([[1, 13, 17, 2]])
    output = model(
        source,
        torch.ones_like(source, dtype=torch.bool),
        torch.tensor([[False, True, True, False]]),
        torch.tensor([[1, 5]]),
        torch.ones(1, 2, dtype=torch.bool),
        target,
        torch.ones_like(target, dtype=torch.bool),
        target,
    )
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("non-finite smoke loss")
    output.loss.backward()
    print(f"side-memory smoke passed; loss={output.loss.detach().item():.6f}")


if __name__ == "__main__":
    main()
