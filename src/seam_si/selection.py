"""Select exactly one generated target per train source using Perl ROUGE-2 F1."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluate_rouge_vietnamese import validate_rouge_install
from evaluate_seam_candidates import score_candidates
from si_common import batches, check_candidates, digest, file_digest, json_line, read_rows, write_json


def best_rouge2(row):
    usable = [candidate for candidate in row["candidates"] if candidate["prediction"].strip()]
    if not usable:
        raise ValueError(f"All 8 candidates are empty at train row {row['row_index']}; source is not dropped")
    best = max(usable, key=lambda candidate: (candidate["rouge"]["rouge2"], -candidate["index"]))
    return {
        "row_index": row["row_index"],
        "line_number": row["line_number"],
        "id": row["id"],
        "text": row["source"],
        "summary": best["prediction"].strip(),
        "reference": row["reference"],
        "selected_index": best["index"],
        "rouge": best["rouge"],
        "candidate_scores": [{"index": item["index"], **item["rouge"]} for item in row["candidates"]],
    }


def score_batch(index, batch, work, rouge_path):
    """Each job owns its Perl temporary directory and atomic cache file."""
    shard = work / f"batch_{index:06d}.jsonl"
    if shard.exists():
        selected = list(read_rows(shard))
        if [item["row_index"] for item in selected] != [item["row_index"] for item in batch]:
            raise ValueError(f"Scoring shard disagrees with candidate order: {shard}")
        return selected
    with tempfile.TemporaryDirectory(prefix="rouge-si-") as temporary:
        temporary = Path(temporary)
        scored = score_candidates(batch, rouge_path, temporary)
        selected = [best_rouge2(row) for row in scored]
        shutil.copyfile(temporary / "rouge_raw.txt", work / f"batch_{index:06d}.rouge.txt")
    shard_tmp = shard.with_suffix(".jsonl.tmp")
    with shard_tmp.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json_line(row))
    shard_tmp.replace(shard)
    return selected


def scored_batches(destination, batch_size, workers, work, rouge_path):
    """Bound queued batches by worker count; publish in original order."""
    jobs = iter(enumerate(batches(read_rows(destination), batch_size)))
    if workers == 1:
        for index, batch in jobs:
            yield score_batch(index, batch, work, rouge_path)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for index, batch in jobs:
            pending.append(pool.submit(score_batch, index, batch, work, rouge_path))
            if len(pending) >= workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def select(args):
    workers = getattr(args, "score_workers", 1)
    if workers < 1:
        raise ValueError("score-workers must be positive")
    destination = args.output_dir / "candidates.jsonl"
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sharding"):
        raise ValueError("Merge prediction shards before selecting the complete SI train set")
    total, seen = int(manifest["total_sources"]), set()
    completion = args.output_dir / "prediction_complete.json"
    if (
        total <= 0
        or not completion.exists()
        or json.loads(completion.read_text(encoding="utf-8")) != {"count": total, "manifest_sha256": digest(manifest)}
    ):
        raise ValueError("Prediction is incomplete or unverified; finish --stage predict first")
    # Never publish a training set from partial or duplicated rollout results.
    for row in read_rows(destination):
        check_candidates(row)
        index = row["row_index"]
        if index in seen or not 0 <= index < total:
            raise ValueError(f"Duplicate/out-of-range candidate row_index: {index}")
        seen.add(index)
    if len(seen) != total:
        raise ValueError(f"Prediction is incomplete: {len(seen)}/{total}; finish --stage predict first")
    rouge_path = args.rouge_path.expanduser().resolve()
    validate_rouge_install(rouge_path)
    identity = {
        "candidates_sha256": file_digest(destination),
        "generation": manifest,
        "selection": "maximum Perl ROUGE-2 F1; lower candidate index breaks ties",
        "rouge_path": str(rouge_path),
        "rouge_script_sha256": file_digest(rouge_path / "ROUGE-1.5.5.pl"),
        "rouge_args": "-a -m -n 2 -d",
        "batch_sources": args.score_batch_size,
    }
    work = args.output_dir / "rouge155"
    work.mkdir(exist_ok=True)
    identity_path = work / "manifest.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Candidate/scoring settings changed; keep original settings or use a new output directory")
    write_json(identity_path, identity)
    started, completed = time.monotonic(), 0
    train_path, audit_path = args.output_dir / "train_si.jsonl", args.output_dir / "selected.jsonl"
    train_tmp, audit_tmp = train_path.with_suffix(".jsonl.tmp"), audit_path.with_suffix(".jsonl.tmp")
    sums, winners = {metric: 0.0 for metric in ("rouge1", "rouge2", "rougeL")}, [0] * 8
    with train_tmp.open("w", encoding="utf-8") as train_stream, audit_tmp.open("w", encoding="utf-8") as audit_stream:
        for selected in scored_batches(destination, args.score_batch_size, workers, work, rouge_path):
            for row in selected:
                train_stream.write(json_line({key: row[key] for key in ("id", "text", "summary")}))
                audit_stream.write(json_line({key: value for key, value in row.items() if key != "text"}))
                winners[row["selected_index"]] += 1
                for metric in sums:
                    sums[metric] += row["rouge"][metric]
            completed += len(selected)
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"[select] {completed}/{total} | elapsed={elapsed / 60:.1f}m | ETA={(total - completed) * elapsed / completed / 60:.1f}m",
                flush=True,
            )
    if completed != total:
        raise RuntimeError("Selected source count differs from full train count")
    train_tmp.replace(train_path)
    audit_tmp.replace(audit_path)
    report = {
        "count": completed,
        "candidates_per_source": 8,
        "selection_metric": "Perl ROUGE-2 F1",
        "score_scale": "0_to_1",
        "selected_mean_rouge": {key: value / total for key, value in sums.items()},
        "selected_index_counts": winners,
        "train_file": str(train_path.resolve()),
        "train_sha256": file_digest(train_path),
        "selection_provenance": identity,
    }
    write_json(args.output_dir / "summary.json", report)
    print(f"[done] {completed} SI training samples: {train_path}", flush=True)
