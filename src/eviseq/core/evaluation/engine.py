"""Stable EviSeq evaluation runtime for complete epoch, best, and last checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import statistics
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import torch

from ..configuration import load_config, resolve_data_path
from ..data.dataset import clean_text, decoder_seed_ids, encode_source, read_jsonl
from ..modeling.architecture import EviSeq as RuntimeModel
from ..training.checkpoint import load_checkpoint
from ..training.engine import _device, _tokenizers
from .generation import generate
from .metrics import task_scores

LOGGER = logging.getLogger("eviseq.evaluation.engine")


def _format_duration(seconds: float) -> str:
    """Render an ETA without exposing fractional seconds in progress logs."""

    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _row_id(row: Dict[str, Any]) -> str:
    value = row.get("id")
    if value is None or str(value).strip() == "":
        raise ValueError("Evaluation resume requires every dataset row to have a non-empty id")
    return str(value)


def _load_resume_records(
    output: Path,
    rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], set[str]]:
    """Read completed JSONL records and discard an incomplete trailing line."""

    if not output.is_file() or output.stat().st_size == 0:
        return [], set()

    row_ids = {_row_id(row) for row in rows}
    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    valid_end = 0
    with output.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            next_end = handle.tell()
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                break
            if not isinstance(record, dict):
                break
            record_id = record.get("id")
            if record_id is None or str(record_id) not in row_ids:
                raise ValueError("Cannot resume evaluation: prediction file contains an unknown or empty id")
            record_id = str(record_id)
            if record_id in seen:
                raise ValueError(f"Cannot resume evaluation: duplicate prediction id {record_id!r}")
            if "prediction" not in record or "reference" not in record:
                raise ValueError("Cannot resume evaluation: every record needs prediction and reference fields")
            records.append(record)
            seen.add(record_id)
            valid_end = next_end

    if valid_end < output.stat().st_size:
        # A process killed during a write may leave a partial final JSON line.
        with output.open("r+b") as handle:
            handle.truncate(valid_end)
    return records, seen


def _diagnostics(predictions: List[str], references: List[str], sources: List[str]) -> Dict[str, float]:
    prediction_lengths = [len(value.split()) for value in predictions]
    reference_lengths = [len(value.split()) for value in references]
    source_lengths = [len(value.split()) for value in sources]
    ratios = [prediction / max(1, reference) for prediction, reference in zip(prediction_lengths, reference_lengths)]
    normalized = [" ".join(value.lower().split()) for value in predictions if value.strip()]
    prefixes = Counter(" ".join(value.split()[:5]) for value in normalized)

    def repeated(text: str) -> float:
        tokens = text.split()
        if len(tokens) < 3:
            return 0.0
        ngrams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
        return 1.0 - len(set(ngrams)) / len(ngrams)

    count = max(1, len(predictions))
    return {
        "num_examples": len(predictions),
        "prediction_words_mean": round(sum(prediction_lengths) / count, 4),
        "reference_words_mean": round(sum(reference_lengths) / count, 4),
        "length_ratio_mean": round(sum(ratios) / count, 6),
        "empty_prediction_rate": round(100 * sum(not value.strip() for value in predictions) / count, 4),
        "too_short_rate": round(100 * sum(value < 0.5 for value in ratios) / count, 4),
        "too_long_rate": round(100 * sum(value > 1.5 for value in ratios) / count, 4),
        "repeated_trigram_rate_mean": round(100 * sum(repeated(value) for value in predictions) / count, 4),
        "source_words_mean": round(sum(source_lengths) / count, 4),
        "unique_prediction_rate": round(100 * len(set(normalized)) / max(1, len(normalized)), 4),
        "dominant_prefix_5gram_rate": round(100 * max(prefixes.values(), default=0) / max(1, len(normalized)), 4),
    }


def _pad_source(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    data: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    encoded = []
    units = []
    sources = []
    references = []
    clean_metadata = bool(data.get("clean_wikihow_metadata", False))
    for row in rows:
        source = clean_text(row["source"], clean_metadata)
        reference = clean_text(row["target"], clean_metadata)
        ids, unit_ids, _ = encode_source(tokenizer, source, data)
        encoded.append(torch.tensor(ids, dtype=torch.long))
        units.append(torch.tensor(unit_ids, dtype=torch.long))
        sources.append(source)
        references.append(reference)
    length = max(value.numel() for value in encoded)
    input_ids = torch.full((len(rows), length), int(tokenizer.pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
    unit_tensor = torch.zeros((len(rows), length), dtype=torch.long)
    for index, (ids, unit_ids) in enumerate(zip(encoded, units)):
        input_ids[index, : ids.numel()] = ids
        attention_mask[index, : ids.numel()] = 1
        unit_tensor[index, : unit_ids.numel()] = unit_ids
    return (
        input_ids.to(device),
        attention_mask.to(device),
        unit_tensor.to(device),
        sources,
        references,
    )


def evaluate(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    max_samples: int = 0,
    resume: bool = False,
) -> Dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    config = load_config(config_path)
    device = _device()
    encoder_tokenizer, decoder_tokenizer = _tokenizers(config)
    model = RuntimeModel(config)
    payload = load_checkpoint(model, checkpoint)
    model.to(device).eval()
    data = config["data"]
    configured_limit = int(config.get("limits", {}).get("max_test_examples", 0))
    effective_limit = int(max_samples) if int(max_samples) > 0 else configured_limit
    rows = read_jsonl(resolve_data_path(data["test_file"], config), max_examples=effective_limit, data_config=data)
    generation = config.get("generation", {})
    batch_size = int(os.environ.get("EVISEQ_EVAL_BATCH_SIZE", generation.get("batch_size", 64)))
    if batch_size <= 0:
        raise ValueError("Evaluation batch size must be positive")
    LOGGER.warning(
        "Evaluation generation batch=%d (resolved_config=%d, runtime_override=%s)",
        batch_size,
        int(generation.get("batch_size", 64)),
        os.environ.get("EVISEQ_EVAL_BATCH_SIZE", "<none>"),
    )
    decoder_seed = decoder_seed_ids(decoder_tokenizer, data)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    resume_records: List[Dict[str, Any]] = []
    processed_ids: set[str] = set()
    if resume:
        resume_records, processed_ids = _load_resume_records(output, rows)

    row_by_id = {_row_id(row): row for row in rows}
    predictions: List[str] = [str(record["prediction"]).strip() for record in resume_records]
    references: List[str] = [str(record["reference"]).strip() for record in resume_records]
    sources: List[str] = [str(row_by_id[str(record["id"])]["source"]) for record in resume_records]
    latencies: List[float] = []
    # Stream completed batches so long evaluations remain observable and a
    # partial prediction file survives an interruption.
    output_handle = output.open("a" if resume else "w", encoding="utf-8")
    generation_started = time.perf_counter()
    bf16 = device.type == "cuda" and bool(config.get("training", {}).get("bf16", True))

    def autocast():
        if bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    pending_rows = [row for row in rows if _row_id(row) not in processed_ids]
    cursor = 0
    active_batch_size = batch_size
    if resume_records:
        resume_message = (
            f"resuming evaluation: {len(resume_records)}/{len(rows)} predictions already present; "
            f"{len(pending_rows)} remaining"
        )
        LOGGER.info(resume_message)
        print(resume_message, flush=True)
    while cursor < len(pending_rows):
        batch_rows = pending_rows[cursor : cursor + active_batch_size]
        input_ids = attention_mask = unit_ids = generated = None
        try:
            input_ids, attention_mask, unit_ids, batch_sources, batch_references = _pad_source(
                batch_rows,
                encoder_tokenizer,
                data,
                device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            began = time.perf_counter()
            with autocast():
                generated = generate(
                    model,
                    input_ids,
                    attention_mask,
                    decoder_seed,
                    unit_ids=unit_ids,
                    max_new_tokens=int(generation.get("max_new_tokens", 256)),
                    min_new_tokens=int(generation.get("min_new_tokens", 16)),
                    eos_token_id=decoder_tokenizer.eos_token_id,
                    pad_token_id=decoder_tokenizer.pad_token_id,
                    repetition_penalty=float(generation.get("repetition_penalty", 1.05)),
                    no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 3)),
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
        except torch.cuda.OutOfMemoryError:
            del input_ids, attention_mask, unit_ids, generated
            gc.collect()
            torch.cuda.empty_cache()
            if active_batch_size <= 1:
                raise
            previous_batch_size = active_batch_size
            active_batch_size = max(1, active_batch_size // 2)
            LOGGER.warning(
                "CUDA OOM at evaluation example %d; retrying with batch=%d (was %d)",
                len(resume_records) + cursor,
                active_batch_size,
                previous_batch_size,
            )
            continue

        if generated is None:
            raise RuntimeError("Evaluation generation returned no token sequences")
        decoded = decoder_tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(value.strip() for value in decoded)
        references.extend(batch_references)
        sources.extend(batch_sources)
        latencies.extend([elapsed / len(batch_rows)] * len(batch_rows))
        for row, prediction, reference in zip(batch_rows, decoded, batch_references):
            output_handle.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "prediction": prediction.strip(),
                        "reference": reference,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        output_handle.flush()
        cursor += len(batch_rows)
        elapsed = time.perf_counter() - generation_started
        completed = len(resume_records) + cursor
        rate = max(0, completed - len(resume_records)) / max(elapsed, 1e-9)
        remaining = max(0, len(rows) - completed)
        eta = remaining / max(rate, 1e-9)
        progress = (
            f"evaluation progress: {completed}/{len(rows)} predictions written | "
            f"rate={rate:.2f} examples/s | elapsed={_format_duration(elapsed)} | "
            f"eta={_format_duration(eta)}"
        )
        LOGGER.info(progress)
        # Keep long evaluations observable even when an embedding application
        # has already installed a WARNING-only logging handler. The explicit
        # flush is also required for live `tee`/`tail -f` monitoring.
        print(progress, flush=True)
    output_handle.close()
    metrics: Dict[str, Any] = {
        **task_scores(predictions, references, config.get("task", {})),
        **_diagnostics(predictions, references, sources),
        "checkpoint": str(checkpoint),
        "checkpoint_role": payload.get("checkpoint_role"),
        "task_metrics": list(config.get("task", {}).get("metrics", ["rouge"])),
        "resumed_examples": len(resume_records),
        "generated_examples_this_run": len(pending_rows),
        "latency_seconds_mean": round(statistics.mean(latencies), 6) if latencies else 0.0,
        "decode_elapsed_seconds": round(sum(latencies), 3),
        "decode_examples_per_second": round(
            len(pending_rows) / max(1e-9, sum(latencies)) if pending_rows else 0.0,
            6,
        ),
        "encoder_name": config["model"]["encoder_name"],
        "decoder_name": config["model"]["decoder_name"],
        "final_graph": "one_encoder_one_adapter_one_decoder",
    }
    benchmark_root = config.get("benchmark", {})
    benchmark = benchmark_root.get("paper", benchmark_root)
    if all(name in benchmark for name in ("rouge1", "rouge2", "rougeL")) and all(
        name in metrics for name in ("rouge1", "rouge2", "rougeL")
    ):
        metrics["benchmark_name"] = str(benchmark.get("name", "T5Gemma"))
        metrics["gap_to_t5gemma"] = {
            name: round(float(metrics[name]) - float(benchmark[name]), 4) for name in ("rouge1", "rouge2", "rougeL")
        }
        metrics["rouge2_target_reached"] = float(metrics["rouge2"]) >= float(benchmark["rouge2"])
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if "gap_to_t5gemma" in metrics:
        gap_path = output.parent / "t5gemma_gap_report.json"
        gap_path.write_text(
            json.dumps(
                {
                    "candidate_checkpoint": str(checkpoint),
                    "candidate": {name: metrics[name] for name in ("rouge1", "rouge2", "rougeL")},
                    "target_name": metrics["benchmark_name"],
                    "target": {name: float(benchmark[name]) for name in ("rouge1", "rouge2", "rougeL")},
                    "candidate_minus_target": metrics["gap_to_t5gemma"],
                    "rouge2_target_reached": metrics["rouge2_target_reached"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from completed JSONL predictions at --output")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output, args.max_samples, resume=args.resume)


if __name__ == "__main__":
    main()
