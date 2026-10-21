"""Independent GPU hosts for disjoint SI shards, followed by a verified merge."""

from __future__ import annotations

import copy
import json
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from prediction import predict
from si_common import check_candidates, digest, json_line, read_rows, shard_bounds, write_json


def merge_shards(output_dir):
    first_manifest_path = output_dir / "shard0" / "manifest.json"
    if not first_manifest_path.is_file():
        raise ValueError("Shard 0 is missing; finish all prediction shards before merging")
    first_manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    first_sharding = first_manifest.get("sharding", {})
    num_shards = first_sharding.get("num_shards")
    total = first_sharding.get("global_total")
    if not isinstance(num_shards, int) or num_shards < 2:
        raise ValueError("Shard 0 has invalid multi-GPU provenance")
    if not isinstance(total, int) or total < num_shards:
        raise ValueError("Train set must contain at least one source per GPU shard")

    identities = []
    for rank in range(num_shards):
        directory = output_dir / f"shard{rank}"
        if any(
            not (directory / name).is_file()
            for name in ("manifest.json", "prediction_complete.json", "candidates.jsonl")
        ):
            raise ValueError(f"Shard {rank} is incomplete; finish all {num_shards} prediction shards before merging")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        completion = json.loads((directory / "prediction_complete.json").read_text(encoding="utf-8"))
        if completion != {"count": manifest["total_sources"], "manifest_sha256": digest(manifest)}:
            raise ValueError(f"Shard {rank} is incomplete or has inconsistent provenance")
        sharding = manifest.get("sharding", {})
        start, stop = shard_bounds(total, rank, num_shards)
        expected = {
            "rank": rank,
            "num_shards": num_shards,
            "start": start,
            "stop": stop,
            "global_total": total,
        }
        if sharding != expected:
            raise ValueError(f"Invalid shard {rank} range/rank")
        if manifest["total_sources"] != stop - start:
            raise ValueError(f"Invalid shard {rank} count")
        identities.append(manifest)

    baseline = {key: value for key, value in identities[0].items() if key not in ("total_sources", "sharding")}
    for rank, identity in enumerate(identities[1:], start=1):
        other = {key: value for key, value in identity.items() if key not in ("total_sources", "sharding")}
        if baseline != other:
            raise ValueError(f"Shard {rank} used different data, config, teacher, or sampling settings")

    merged_identity = {
        **baseline,
        "total_sources": total,
        "prediction_shards": [{"manifest_sha256": digest(identity), **identity["sharding"]} for identity in identities],
    }
    manifest_path, candidates_path = output_dir / "manifest.json", output_dir / "candidates.jsonl"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != merged_identity:
        raise ValueError("Parent output belongs to a different experiment; use a fresh --output-dir")
    if not manifest_path.exists() and candidates_path.exists() and candidates_path.stat().st_size:
        raise ValueError("Parent candidates have no manifest; use a fresh --output-dir")

    completion_path = output_dir / "prediction_complete.json"
    completion_path.unlink(missing_ok=True)
    temporary, seen = candidates_path.with_suffix(".jsonl.tmp"), set()
    with temporary.open("w", encoding="utf-8") as stream:
        for rank, manifest in enumerate(identities):
            start, stop = shard_bounds(total, rank, num_shards)
            count = 0
            for row in read_rows(output_dir / f"shard{rank}" / "candidates.jsonl"):
                check_candidates(row)
                index = row["row_index"]
                record = {key: row[key] for key in ("row_index", "line_number", "id", "source", "reference")}
                if not start <= index < stop or index in seen or row["record_sha256"] != digest(record):
                    raise ValueError(f"Duplicate, out-of-range, or corrupted row {index} in shard {rank}")
                seen.add(index)
                stream.write(json_line(row))
                count += 1
            if count != manifest["total_sources"]:
                raise ValueError(f"Incomplete candidate file in shard {rank}: {count}/{manifest['total_sources']}")
    if len(seen) != total:
        raise ValueError(f"Missing merged rows: {len(seen)}/{total}")
    write_json(manifest_path, merged_identity)
    temporary.replace(candidates_path)
    write_json(completion_path, {"count": total, "manifest_sha256": digest(merged_identity)})
    print(
        f"[merge] {total} unique sources from {num_shards} GPU shards, 8 candidates/source: {candidates_path}",
        flush=True,
    )


def predict_multi_gpu(args):
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",")]
    if not gpu_ids or any(not gpu for gpu in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpus must list one or more distinct GPU IDs")
    num_shards = len(gpu_ids)
    if args.server_port + num_shards - 1 > 65535:
        raise ValueError(f"{num_shards} hosts from port {args.server_port} exceed the valid TCP port range")

    if num_shards == 1:
        args.gpu_id = gpu_ids[0]
        print(f"[SI] GPU {gpu_ids[0]} -> full train set | port={args.server_port}", flush=True)
        return predict(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = previous.get("prediction_shards")
        if not shards:
            raise ValueError("This output belongs to a single-GPU run; choose a new multi-GPU --output-dir")
        if len(shards) != num_shards:
            raise ValueError("This output was created with a different GPU count; choose a new --output-dir")
    elif (args.output_dir / "candidates.jsonl").exists() and (args.output_dir / "candidates.jsonl").stat().st_size:
        raise ValueError("Parent candidates have no manifest; choose a new --output-dir")

    cancel, lock, hosts = threading.Event(), threading.Lock(), []

    def stop(process):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def register(process):
        with lock:
            hosts.append(process)
        if cancel.is_set():
            stop(process)

    children = []
    for rank, gpu_id in enumerate(gpu_ids):
        child = copy.copy(args)
        child.output_dir = args.output_dir / f"shard{rank}"
        child.gpu_id, child.shard_rank, child.num_shards = gpu_id, rank, num_shards
        child.server_port = args.server_port + rank
        child.cancel_event, child.on_server_started = cancel, register
        if args.server_log:
            child.server_log = args.server_log.with_name(f"{args.server_log.stem}.shard{rank}{args.server_log.suffix}")
        children.append(child)
        print(
            f"[SI] GPU {gpu_id} -> rows shard{rank}/{num_shards} | port={child.server_port} | output={child.output_dir}",
            flush=True,
        )

    executor = ThreadPoolExecutor(max_workers=num_shards)
    try:
        futures = [executor.submit(predict, child) for child in children]
        for future in as_completed(futures):
            future.result()
    except BaseException:
        cancel.set()
        with lock:
            active_hosts = list(hosts)
        for process in active_hosts:
            stop(process)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    merge_shards(args.output_dir)


def predict_two_gpu(args):
    """Backward-compatible name for callers of the original two-GPU coordinator."""
    return predict_multi_gpu(args)
