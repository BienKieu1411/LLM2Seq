#!/usr/bin/env python3
"""Evaluate length-conditioned checkpoints; a test gold-length oracle is opt-in."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT.parent / "eviseq_new", ROOT.parent / "seam_grpo", ROOT):
    sys.path.insert(0, str(path))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import contextlib
import json

import torch
import torch.distributed as dist
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import _tokenizers
from eviseq_afmr.training.checkpoint import load_checkpoint
from seam_length_grpo.config import load
from seam_length_grpo.data import LengthCollator, LengthDataset, prepare_eval
from seam_length_grpo.evaluation import predict, summarize
from seam_length_grpo.policy import validate_budgets
from seam_length_grpo.reward import LengthReward
from torch.utils.data import DataLoader, Subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="length_grpo_config.yaml saved by the new trainer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--input", help="Optional prepared JSONL override for the chosen split")
    parser.add_argument("--target-words", type=int, help="External request applied to every row")
    parser.add_argument(
        "--oracle-reference-length", action="store_true", help="Explicit matched-reference-length evaluation"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.target_words is not None and args.oracle_reference_length:
        parser.error("Choose external --target-words or --oracle-reference-length")
    output = Path(args.output).resolve()
    metrics_path = output.with_suffix(".metrics.json")
    if not args.overwrite:
        for artifact in (output, metrics_path):
            if artifact.exists():
                raise FileExistsError(f"Refusing to replace {artifact}; use --overwrite for an intentional rerun")
    settings = load(args.config)
    cfg = settings["training"]
    base = prepare_eval(settings)
    if args.input:
        base["data"][f"{args.split}_file"] = str(Path(args.input).expanduser().resolve())
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(
        args.device or (f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        tokenizers = _tokenizers(base)
        length = settings["length_control"]
        # External W overrides input metadata without reading test references for the request.
        if args.target_words is not None:
            if not 1 <= args.target_words <= length["max_requested_words"]:
                raise ValueError("--target-words is outside the configured range")
            records = LengthDataset(
                base["data"][f"{args.split}_file"],
                base["data"],
                length,
                derive_words=False,
                requested_words=args.target_words,
            )
        else:
            records = LengthDataset(
                base["data"][f"{args.split}_file"],
                base["data"],
                length,
                derive_words=args.oracle_reference_length,
                force_reference_length=args.oracle_reference_length,
            )
        collator = LengthCollator(
            tokenizers,
            base["data"],
            length,
            include_targets=False,
            grounded_copy=bool(base["decoder"].get("grounded_copy", {}).get("enabled", False)),
        )
        loader = DataLoader(
            Subset(records, range(rank, len(records), world)),
            batch_size=args.batch_size or cfg["validation_batch_size"],
            collate_fn=collator,
        )
        model = EviSeqAFMR(base).float().to(device).eval()
        validate_budgets(model, collator, cfg["max_new_tokens"])
        load_checkpoint(args.checkpoint, model, config=base, restore_rng=False)
        dtype = base["model"].get("compute_dtype", base["model"].get("dtype", "float32"))

        def autocast():
            return (
                torch.autocast("cuda", dtype=getattr(torch, dtype))
                if device.type == "cuda" and dtype in {"float16", "bfloat16"}
                else contextlib.nullcontext()
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        shard = output if world == 1 else output.with_name(f"{output.stem}.shard{rank}.jsonl")
        metrics = predict(
            model,
            loader,
            tokenizers[1],
            cfg,
            base,
            LengthReward(settings["reward"]),
            device,
            autocast,
            shard,
            overwrite=args.overwrite,
        )
        if world > 1:
            dist.barrier()
            if rank == 0:
                pieces = [
                    [
                        json.loads(line)
                        for line in output.with_name(f"{output.stem}.shard{i}.jsonl").read_text().splitlines()
                    ]
                    for i in range(world)
                ]
                merged = [
                    row
                    for offset in range(max(map(len, pieces), default=0))
                    for piece in pieces
                    if offset < len(piece)
                    for row in [piece[offset]]
                ]
                with output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
                    for row in merged:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                metrics = summarize(merged)
        if rank == 0:
            with metrics_path.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(metrics, ensure_ascii=False))
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
