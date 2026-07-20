"""Compile a learned top-K gate checkpoint into a fixed single-pass config."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from .config import dump_config


def export_policy(checkpoint_path: str, output_path: str, output_dir: str | None = None) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = deepcopy(payload["config"])
    state = payload["model_state_dict"]
    gate_names = [name for name in state if name.endswith("policy.gate_logits")]
    if len(gate_names) != 1:
        raise RuntimeError(f"Expected exactly one gate_logits tensor, got {gate_names}")
    logits = state[gate_names[0]].float()
    mask = config.get("mask", {}) or {}
    budget = int(mask.get("budget_groups", 2))
    selected = sorted(torch.topk(logits, k=budget, largest=True).indices.tolist())
    config["mask"] = {
        **mask,
        "mode": "fixed",
        "fixed_strategy": "custom",
        "selected_groups": selected,
    }
    if output_dir is not None:
        config.setdefault("experiment", {})["output_dir"] = output_dir
    config.setdefault("_meta", {})["compiled_from"] = str(Path(checkpoint_path).resolve())
    dump_config(config, output_path)
    print(f"selected_groups={selected}")
    print(f"wrote={Path(output_path).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    export_policy(args.checkpoint, args.output, args.output_dir)


if __name__ == "__main__":
    main()
