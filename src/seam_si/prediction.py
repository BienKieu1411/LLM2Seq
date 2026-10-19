"""Bounded HTTP rollout of eight candidates for every train source."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from itertools import islice

from evaluate_rouge_vietnamese_vllm import SEAMHTTPClient, managed_seam_server
from si_common import check_candidates, digest, file_digest, json_line, read_rows, records, shard_bounds, write_json


def predict(args):
    from eviseq_afmr.config import load_config, resolve_path

    config = load_config(args.config, train_only=True)
    train = (args.train_file or resolve_path(config["data"]["train_file"], config)).expanduser().resolve()
    train_stat = (train.stat().st_size, train.stat().st_mtime_ns)
    global_total = sum(1 for _ in records(train, config["data"]))
    if not global_total:
        raise ValueError("Training file is empty")
    rank, count = getattr(args, "shard_rank", 0), getattr(args, "num_shards", 1)
    start, stop = shard_bounds(global_total, rank, count)
    total = stop - start
    if total == 0:
        raise ValueError("Train has fewer sources than shards; use fewer GPUs")
    tag = f"[predict shard={rank}/{count}]" if count > 1 else "[predict]"
    train_sha = file_digest(train)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The host must reserve the full configured encoder maximum, not a request cap.
    args.max_num_batched_tokens = max(args.max_num_batched_tokens, int(config["data"]["max_source_length"]))
    if args.start_server:
        if args.attention_backend in ("FLASH_ATTN", "TRITON_ATTN"):
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        context = managed_seam_server(args, args.output_dir, args.config.resolve(), args.checkpoint.resolve())
    else:
        context = nullcontext(args.base_url)
    with context as base_url:
        client = SEAMHTTPClient(base_url, args.api_key, timeout=args.timeout, retries=args.retries)
        info = client.request("/seam/info")
        if info.get("implementation") != "SEAMForQwen3" or not info.get("fingerprint"):
            raise ValueError("Endpoint is not the SEAM HTTP host with provenance support")
        if args.server_model not in [row.get("id") for row in client.request("/models").get("data", [])]:
            raise ValueError(f"Host does not expose model {args.server_model!r}")
        if args.checkpoint and info.get("checkpoint", {}).get("path") != str(args.checkpoint.expanduser().resolve()):
            raise ValueError("Host checkpoint differs from --checkpoint")
        expected_prompts = {
            "encoder_prompt": config["data"].get("encoder_prefix", ""),
            "encoder_task_marker": config["data"].get("encoder_task_marker", ""),
            "decoder_prompt": config["data"].get("decoder_prompt", ""),
            "decoder_prefix": config["data"].get("decoder_prefix", ""),
            "decoder_chat_template": bool(config["data"].get("decoder_chat_template", False)),
        }
        if info.get("prompt_defaults") != expected_prompts:
            raise ValueError("Host encoder/decoder prompts differ from --config")
        source_limit = args.max_input_tokens or int(info["max_input_tokens"])
        output_limit = args.max_new_tokens or int(info["generation_defaults"]["max_tokens"])
        if not 0 < source_limit <= int(info["max_input_tokens"]) or output_limit < 1:
            raise ValueError("Invalid input/output budget")
        defaults = {
            "model": args.server_model,
            "n": 8,
            "stream": False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": -1 if args.top_k == 0 else args.top_k,
            "max_input_tokens": source_limit,
            "max_tokens": output_limit,
            "min_tokens": 0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }
        identity = {
            "train_file": str(train),
            "train_sha256": train_sha,
            "total_sources": total,
            "config_sha256": digest(config),
            "server_fingerprint": info["fingerprint"],
            "server": info,
            "request_defaults": defaults,
            "seed": args.seed,
        }
        if count > 1:
            identity["sharding"] = {
                "rank": rank,
                "num_shards": count,
                "start": start,
                "stop": stop,
                "global_total": global_total,
            }
        manifest, destination = args.output_dir / "manifest.json", args.output_dir / "candidates.jsonl"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("Data/model/generation changed; choose a new --output-dir")
        if not manifest.exists() and destination.exists() and destination.stat().st_size:
            raise ValueError("Existing candidates have no manifest; choose a new --output-dir")
        write_json(manifest, identity)
        saved = {}
        for row in read_rows(destination, recover_tail=True):
            check_candidates(row)
            index = row["row_index"]
            if index in saved or not start <= index < stop:
                raise ValueError(f"Duplicate/out-of-range saved row_index: {index}")
            saved[index] = row["record_sha256"]

        def pending():
            for row in islice(records(train, config["data"]), start, stop):
                if row["row_index"] not in saved:
                    yield row
                elif saved[row["row_index"]] != row["record_sha256"]:
                    raise ValueError("Saved source/reference disagrees with training data")

        def request(row):
            # Gold is used ONLY later by the Perl scorer, never sent to the host.
            body = {**defaults, "prompt": row["source"], "seed": args.seed + row["row_index"]}
            result = client.request("/completions", body, fingerprint=info["fingerprint"])
            choices = result.get("choices")
            if not isinstance(choices, list) or len(choices) != 8:
                raise RuntimeError(f"Expected 8 HTTP choices at row {row['row_index']}")
            candidates = [
                {"index": choice["index"], "prediction": choice["text"], "finish_reason": choice.get("finish_reason")}
                for choice in sorted(choices, key=lambda choice: choice["index"])
            ]
            output = {**row, "candidates": candidates}
            check_candidates(output)
            return output

        iterator, generated, started = iter(pending()), 0, time.monotonic()
        print(
            f"{tag} train={total} | rows=[{start},{stop}) | reused={len(saved)} | n=8 | concurrency={args.batch_size}",
            flush=True,
        )
        with destination.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.batch_size) as pool:

            def persist(future):
                nonlocal generated
                stream.write(json_line(future.result()))
                stream.flush()
                generated += 1

            futures = set()
            for _ in range(args.batch_size):
                row = next(iterator, None)
                if row is not None:
                    futures.add(pool.submit(request, row))
            while futures:
                done, futures = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                cancelled = getattr(args, "cancel_event", None)
                errors = []
                for future in done:
                    try:
                        persist(future)
                    except Exception as exc:
                        errors.append(exc)
                if cancelled is not None and cancelled.is_set():
                    errors.append(RuntimeError("Other GPU failed or SI was interrupted; resume saved shards later"))
                if errors:
                    # Preserve successful requests already in flight before stopping.
                    for future in futures:
                        try:
                            persist(future)
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"SI prediction failed; successful rows saved for resume: {errors[0]}"
                    ) from errors[0]
                if not done:
                    continue
                if generated == 1 or generated % args.log_every == 0 or generated + len(saved) == total:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = generated / elapsed
                    print(
                        f"{tag} {generated + len(saved)}/{total} | ex/s={rate:.2f} "
                        f"| elapsed={elapsed / 60:.1f}m | ETA={(total - len(saved) - generated) / rate / 60:.1f}m",
                        flush=True,
                    )
                for _ in done:
                    row = next(iterator, None)
                    if row is not None:
                        futures.add(pool.submit(request, row))
        if len(saved) + generated != total or train_stat != (train.stat().st_size, train.stat().st_mtime_ns):
            raise RuntimeError("Train data changed during prediction or some sources are missing")
        write_json(args.output_dir / "prediction_complete.json", {"count": total, "manifest_sha256": digest(identity)})
        print(f"{tag} Complete: {destination}", flush=True)
