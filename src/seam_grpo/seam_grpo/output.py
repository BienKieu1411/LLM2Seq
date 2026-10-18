"""Explicit output reset, synchronized before any GRPO rank writes artifacts."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import torch.distributed as dist
from accelerate import PartialState


def reset_output(settings: dict, base: dict, *, use_cpu: bool) -> None:
    state = PartialState(cpu=use_cpu)
    error = [None]
    if state.is_main_process:
        try:
            output = Path(settings["output_dir"]).resolve()
            protected = [
                Path(__file__).resolve(),
                Path.cwd().resolve(),
                Path.home().resolve(),
                Path(settings["base_config"]).resolve(),
                Path(settings["sft_checkpoint"]).resolve(),
            ]
            protected.extend(
                Path(value).resolve() for key, value in base["data"].items() if key.endswith("_file") and value
            )
            protected.extend(
                Path(base["model"][key]).resolve()
                for key in ("encoder_name", "decoder_name")
                if base["model"][key] != "__tiny__"
            )
            if any(path == output or output in path.parents for path in protected):
                raise ValueError(
                    "Refusing to overwrite a directory containing code, configs, checkpoints or input data"
                )
            if output.exists():
                if not output.is_dir():
                    raise ValueError(f"output_dir is not a directory: {output}")
                logging.getLogger(__name__).warning("Removing previous GRPO artifacts from %s", output)
                shutil.rmtree(output)
            output.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if state.num_processes > 1:
        # Broadcasting also keeps every rank behind rank 0's reset. Propagate
        # failures to peers instead of leaving them waiting at a barrier.
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"GRPO output reset failed: {error[0]}")
