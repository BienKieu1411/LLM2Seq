#!/usr/bin/env python3
"""Generate eight SEAM candidates for the entire train set, then select by ROUGE-2."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import si_common  # Set local source-package paths before importing shared evaluators.
from evaluate_rouge_vietnamese import ROUGE_PATH


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "predict", "merge", "select"), default="all")
    parser.add_argument("--config", type=Path, help="Resolved SFT/SEAM config; required for prediction")
    parser.add_argument("--train-file", type=Path, help="Default: data.train_file from --config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", dest="server_model", default="seam")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument(
        "--checkpoint", type=Path, help="Auto host: local .pt; existing host: verify checkpoint identity"
    )
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--gpus", help="Two independent host GPUs, e.g. 0,1; requires --start-server")
    parser.add_argument("--shard-rank", type=int, default=0, help="Worker-only contiguous shard rank")
    parser.add_argument("--num-shards", type=int, default=1, help="Worker-only shard count")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--batch-size", "--concurrency", type=int, default=8, help="Concurrent sources, 8 candidates each"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, help="Default: host's configured source limit")
    parser.add_argument("--max-new-tokens", type=int, help="Default: host's configured output limit")
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--encoder-max-batch-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--attention-backend", choices=("auto", "FLASH_ATTN", "TRITON_ATTN", "FLASHINFER"), default="FLASH_ATTN"
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--rouge-path", type=Path, default=ROUGE_PATH)
    parser.add_argument(
        "--score-batch-size", type=int, default=128, help="Sources per Perl job; RAM/temp files remain bounded"
    )
    args = parser.parse_args()
    if args.stage in ("all", "predict") and args.config is None:
        parser.error("--config is required for prediction")
    if args.stage in ("all", "predict") and args.start_server and args.checkpoint is None:
        parser.error("--start-server requires --checkpoint")
    if args.num_shards < 1 or not 0 <= args.shard_rank < args.num_shards:
        parser.error("Require num-shards >= 1 and 0 <= shard-rank < num-shards")
    if args.num_shards > 1 and (args.stage != "predict" or args.gpus):
        parser.error("Shard workers require --stage predict and no --gpus; merge before selection")
    if args.gpus:
        gpu_ids = [value.strip() for value in args.gpus.split(",")]
        if len(gpu_ids) != 2 or not all(gpu_ids) or len(set(gpu_ids)) != 2:
            parser.error("--gpus must name two distinct GPUs, e.g. 0,1")
        if args.stage in ("all", "predict") and not args.start_server:
            parser.error("--gpus requires --start-server")
    if (
        min(
            args.batch_size,
            args.score_batch_size,
            args.max_num_seqs,
            args.max_num_batched_tokens,
            args.encoder_max_batch_tokens,
            args.log_every,
        )
        < 1
        or args.retries < 0
    ):
        parser.error("Batch/token/log counts must be positive; retries must be nonnegative")
    if not math.isfinite(args.temperature) or args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("Require temperature > 0, 0 < top-p <= 1, top-k >= 0")
    if any(value is not None and value < 1 for value in (args.max_input_tokens, args.max_new_tokens)):
        parser.error("Input/output budgets must be positive")
    if not 0 < args.gpu_memory_utilization < 1 or min(args.timeout, args.startup_timeout) <= 0:
        parser.error("Require 0 < gpu-memory-utilization < 1 and positive timeouts")
    if not 1 <= args.server_port <= 65535:
        parser.error("--server-port must be in [1, 65535]")
    if args.gpus and args.server_port == 65535:
        parser.error("Two hosts need --server-port <= 65534; second host uses port+1")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main():
    args = parse_args()
    if args.stage in ("all", "predict"):
        if args.gpus:
            from parallel import predict_two_gpu

            predict_two_gpu(args)
        else:
            from prediction import predict

            predict(args)
    if args.stage == "merge":
        from parallel import merge_shards

        merge_shards(args.output_dir)
    if args.stage in ("all", "select"):
        from selection import select

        select(args)


if __name__ == "__main__":
    main()
