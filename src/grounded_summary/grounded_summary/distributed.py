"""Process-group lifecycle and rank-zero I/O for torchrun."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Callable, TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def is_main_process() -> bool:
    return rank() == 0


def run_on_main(function: Callable[[], T]) -> T | None:
    """Run rank-zero filesystem work and propagate failures to every rank."""
    if world_size() == 1:
        return function()
    outcome: list[str | None] = [None]
    result: T | None = None
    if is_main_process():
        try:
            result = function()
        except Exception as error:  # pragma: no cover - exercised by torchrun
            outcome[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(outcome, src=0)
    if outcome[0] is not None:
        raise RuntimeError(f"Rank-zero operation failed: {outcome[0]}")
    return result


@contextmanager
def training_process_group(device: str | None = None):
    """Initialize and tear down the env:// group used by torchrun.

    ``--device cuda`` selects ``LOCAL_RANK`` for each worker. CPU/Gloo is kept
    available for deterministic offline smoke tests.
    """
    requested_world = int(os.environ.get("WORLD_SIZE", "1"))
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    distributed = requested_world > 1 or world_size() > 1
    if distributed and selected.type not in {"cuda", "cpu"}:
        raise ValueError("Distributed training supports CUDA/NCCL or CPU/Gloo")
    if selected.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if distributed and selected.index is not None and selected.index != local_rank:
            raise ValueError("Use --device cuda with torchrun; each worker uses its LOCAL_RANK GPU")
        selected = torch.device("cuda", local_rank if distributed else selected.index or 0)
        torch.cuda.set_device(selected)
    created = False
    logger = logging.getLogger("grounded_summary")
    previous_level = logger.level
    try:
        if requested_world > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if selected.type == "cuda" else "gloo",
                init_method="env://",
                timeout=timedelta(minutes=30),
            )
            created = True
        if not is_main_process():
            logger.setLevel(logging.WARNING)
        yield selected
    finally:
        logger.setLevel(previous_level)
        # Do not barrier during cleanup: an exception on another worker must
        # not turn into a deadlock while the process group is being destroyed.
        if created:
            dist.destroy_process_group()
