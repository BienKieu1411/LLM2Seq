#!/usr/bin/env python3
"""Probe bridge routes inside one AFMR checkpoint.

The counterfactuals in this script reuse the trained weights.  They are useful
for checking whether a checkpoint actually uses the bridge, but they are not
replacement retraining ablations and cannot establish a ROUGE improvement.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import torch

# A local path alone does not guarantee that a custom Transformers model will
# avoid Hub lookups for missing code or metadata.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eviseq_afmr.config import load_config  # noqa: E402
from eviseq_afmr.data.copy_alignment import COPY_INPUT_KEYS  # noqa: E402
from eviseq_afmr.modeling.model import EviSeqAFMR  # noqa: E402
from eviseq_afmr.runtime import build_loaders  # noqa: E402
from eviseq_afmr.training.checkpoint import load_checkpoint  # noqa: E402

_MODEL_TENSOR_KEYS = {
    "input_ids",
    "attention_mask",
    "source_content_mask",
    "decoder_prompt_ids",
    "decoder_prompt_mask",
    "decoder_input_ids",
    "decoder_attention_mask",
    "labels",
    "source_salience_labels",
    "source_salience_mask",
    *COPY_INPUT_KEYS,
}
_VARIANTS = (
    "full",
    "zero_source_bias",
    "zero_depth_feature_residuals",
    "zero_source_bias_and_residuals",
)


def _local_model_path(config: dict[str, Any], name: str) -> Path | None:
    if name == "__tiny__":
        return None
    path = Path(name).expanduser()
    if not path.is_absolute():
        config_path = config.get("_meta", {}).get("config_path")
        candidates = [Path.cwd() / path]
        if config_path:
            candidates.insert(0, Path(config_path).parent / path)
        path = next((candidate.resolve() for candidate in candidates if candidate.is_dir()), path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Bridge probe refuses a non-local {name!r}; point the resolved config at a local model directory"
        )
    return path


def _assert_local_backbones(config: dict[str, Any]) -> None:
    _local_model_path(config, str(config["model"]["encoder_name"]))
    _local_model_path(config, str(config["model"]["decoder_name"]))


def _zero_module_output(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError("AFMR residual probe expected a tensor module output")
    return torch.zeros_like(output)


@contextmanager
def _counterfactual(model: EviSeqAFMR, variant: str) -> Iterator[list[str]]:
    """Temporarily disable selected routes without changing checkpoint weights."""

    if variant not in _VARIANTS:
        raise ValueError(f"Unknown bridge probe variant: {variant}")
    zero_prior = variant in {"zero_source_bias", "zero_source_bias_and_residuals"}
    zero_residuals = variant in {"zero_depth_feature_residuals", "zero_source_bias_and_residuals"}
    handles = []
    hooked: list[str] = []
    if zero_residuals:
        for name in ("depth_out", "feature_up"):
            module = getattr(model.bridge, name, None)
            if module is not None:
                handles.append(module.register_forward_hook(_zero_module_output))
                hooked.append(name)

    original = model.encode_source
    if zero_prior:

        def patched_encode_source(*args: Any, **kwargs: Any):
            state = original(*args, **kwargs)
            copy_state = state.copy_state
            if copy_state is not None:
                copy_state = replace(copy_state, bias=torch.zeros_like(copy_state.bias))
            return replace(state, source_bias=torch.zeros_like(state.source_bias), copy_state=copy_state)

        model.__dict__["encode_source"] = patched_encode_source
    try:
        yield hooked
    finally:
        if zero_prior:
            model.__dict__.pop("encode_source", None)
        for handle in handles:
            handle.remove()


def _model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in _MODEL_TENSOR_KEYS and isinstance(value, torch.Tensor)
    }


def _prior_update(stats: dict[str, float], state: Any) -> None:
    valid = state.content_mask.bool() & state.memory_mask.bool()
    values = state.source_bias.float()[valid]
    if values.numel() == 0:
        return
    stats["sum"] += float(values.sum())
    stats["sum_abs"] += float(values.abs().sum())
    stats["sum_sq"] += float(values.square().sum())
    stats["max_abs"] = max(stats["max_abs"], float(values.abs().max()))
    stats["count"] += float(values.numel())


def _prior_result(stats: dict[str, float]) -> dict[str, float | int]:
    count = max(1.0, stats["count"])
    return {
        "content_tokens": int(stats["count"]),
        "mean": stats["sum"] / count,
        "mean_abs": stats["sum_abs"] / count,
        "rms": (stats["sum_sq"] / count) ** 0.5,
        "max_abs": stats["max_abs"],
    }


def _empty_prior_stats() -> dict[str, float]:
    return {"sum": 0.0, "sum_abs": 0.0, "sum_sq": 0.0, "max_abs": 0.0, "count": 0.0}


def probe_model(model: EviSeqAFMR, loader, device: torch.device) -> dict[str, Any]:
    """Return token-weighted CE and source-prior statistics for each route."""

    model.eval()
    results: dict[str, Any] = {}
    with torch.no_grad():
        for variant in _VARIANTS:
            loss_sum = 0.0
            token_total = 0
            prior_stats = _empty_prior_stats()
            with _counterfactual(model, variant) as hooked:
                for batch in loader:
                    inputs = _model_inputs(batch, device)
                    output = model(**inputs, return_logits=False)
                    tokens = int(inputs["labels"][:, 1:].ne(-100).sum())
                    loss_sum += float(output.loss_ce) * tokens
                    token_total += tokens
                    _prior_update(prior_stats, output.bridge)
            results[variant] = {
                "ce": loss_sum / max(1, token_total),
                "target_tokens": token_total,
                "prior": _prior_result(prior_stats),
                "hooked_modules": hooked,
            }
    baseline = float(results["full"]["ce"])
    for result in results.values():
        result["delta_ce_vs_full"] = float(result["ce"]) - baseline
    return results


def run_probe(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    batch_size: int | None = None,
    max_examples: int = 64,
    device: str | None = None,
) -> dict[str, Any]:
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    config = load_config(config_path)
    _assert_local_backbones(config)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    inference_config = copy.deepcopy(config)
    if selected_device.type == "cuda":
        inference_config["model"]["dtype"] = config["model"].get(
            "compute_dtype", config["model"].get("dtype", "float32")
        )
    loaders = build_loaders(
        config,
        split="validation",
        batch_size_override=batch_size,
        max_validation_examples=max_examples,
    )
    model = EviSeqAFMR(inference_config).to(selected_device)
    metadata = load_checkpoint(checkpoint_path, model, config=config, restore_rng=False)
    results = probe_model(model, loaders["validation"], selected_device)
    contextual_value_active = model.bridge.contextual_value is not None
    return {
        "scope": "within_checkpoint_counterfactual",
        "note": "These are teacher-forced CE counterfactuals from one checkpoint; they are not retrained ablations or ROUGE evidence. Residual variants disable depth/feature only.",
        "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": metadata.get("epoch"),
        "checkpoint_step": metadata.get("step"),
        "split": "validation",
        "examples": len(loaders["validation"].dataset),
        "batch_size": int(batch_size or config["training"].get("validation_batch_size", 4)),
        "contextual_value_still_active": contextual_value_active,
        "results": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved AFMR YAML used to train the checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Local AFMR checkpoint, for example last.pt")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=64, help="Validation examples; default is 64")
    parser.add_argument("--device", default=None, help="torch device, for example cuda:0 or cpu")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_probe(
        args.config,
        args.checkpoint,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        device=args.device,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
