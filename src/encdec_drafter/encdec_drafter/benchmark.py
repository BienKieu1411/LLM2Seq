from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .data import read_jsonl, source_text
from .drafter import load_drafter
from .generation import GreedySpeculativeGenerator


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile / 100.0 * len(ordered)) - 1))
    return ordered[index]


def _summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def profile_t5gemma2(adapter, config: Dict[str, Any], row: Dict[str, Any], row_index: int = 0) -> Dict[str, Any]:
    device = adapter.device
    text = source_text(row, config)
    started = time.perf_counter()
    input_ids, attention_mask = adapter.tokenize_source(text)
    tokenization_seconds = time.perf_counter() - started

    _sync(device)
    encoder_start = time.perf_counter()
    source = adapter.encode_ids(input_ids, attention_mask)
    _sync(device)
    encoder_seconds = time.perf_counter() - encoder_start

    cache = adapter.new_cache()
    _sync(device)
    first_start = time.perf_counter()
    first_logits, first_hidden = adapter.forward_decoder(source, [adapter.start_token_id], cache)
    _sync(device)
    first_step_seconds = time.perf_counter() - first_start
    first_token = int(first_logits[0, -1].argmax().item())

    steady_start = time.perf_counter()
    adapter.forward_decoder(source, [first_token], cache)
    _sync(device)
    steady_step_seconds = time.perf_counter() - steady_start
    return {
        "row_index": row_index,
        "source_id": str(row.get(config["data"].get("id_field", "id"), row_index)),
        "source_tokens": int(attention_mask[0].sum().item()),
        "decoder_width": adapter.decoder_width,
        "encoder_width": adapter.encoder_width,
        "target_parameters": sum(parameter.numel() for parameter in adapter.model.parameters()),
        "tokenization_seconds": tokenization_seconds,
        "encoder_seconds": encoder_seconds,
        "first_decoder_step_seconds": first_step_seconds,
        "first_decoder_step_note": "includes lazy per-layer source K/V materialization",
        "steady_decoder_step_seconds": steady_step_seconds,
        "attention_implementation": adapter.attention_implementation,
        "target_cache_bytes_after_two_steps": _cache_bytes(cache.cache),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": str(next(adapter.model.parameters()).dtype),
        "target_hidden_shape": list(source.hidden_states.shape),
        "first_hidden_shape": list(first_hidden.shape),
    }


def _cache_bytes(cache) -> int:
    total = 0
    for component in (cache.self_attention_cache, cache.cross_attention_cache):
        for layer in component.layers:
            for tensor_name in ("keys", "values"):
                tensor = getattr(layer, tensor_name, None)
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
    return int(total)


def benchmark_runtime(
    adapter,
    config: Dict[str, Any],
    input_path: str | Path,
    limit: int,
    repeats: int,
    warmup: int,
    draft_checkpoint: Optional[str | Path],
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    rows = list(read_jsonl(input_path, limit=limit))
    if not rows:
        raise ValueError(f"No benchmark rows found in {input_path}")
    drafter = load_drafter(draft_checkpoint, adapter, config) if draft_checkpoint else None
    if drafter is None:
        raise ValueError("Pass --draft-checkpoint to compare native decoding with speculative decoding")
    generator = GreedySpeculativeGenerator(adapter, config, drafter)

    for row in rows[: min(warmup, len(rows))]:
        text = source_text(row, config)
        ids, mask = adapter.tokenize_source(text)
        adapter.native_generate(ids, mask)
        source = adapter.encode_ids(ids, mask)
        generator.generate_source(source)
    _sync(adapter.device)
    if adapter.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(adapter.device)

    request_records: List[Dict[str, Any]] = []
    native_peak_allocated = 0
    native_peak_reserved = 0
    speculative_peak_allocated = 0
    speculative_peak_reserved = 0
    for row_index, row in enumerate(rows):
        text = source_text(row, config)
        input_ids, attention_mask = adapter.tokenize_source(text)
        native_times: List[float] = []
        speculative_times: List[float] = []
        parity: List[bool] = []
        native_lengths: List[int] = []
        speculative_lengths: List[int] = []
        cycle_stats: List[Dict[str, Any]] = []
        output_diagnostics: List[Dict[str, Any]] = []
        mismatch_diagnostics: List[Dict[str, Any]] = []
        for repeat in range(repeats):
            order = ("native", "speculative") if (row_index + repeat) % 2 == 0 else ("speculative", "native")
            outputs: Dict[str, List[int]] = {}
            for route in order:
                _sync(adapter.device)
                if adapter.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(adapter.device)
                start = time.perf_counter()
                if route == "native":
                    output = adapter.native_generate(input_ids, attention_mask)
                    output_ids = [int(value) for value in output.detach().cpu().tolist()]
                else:
                    source = adapter.encode_ids(input_ids, attention_mask)
                    result = generator.generate_source(
                        source, str(row.get(config["data"].get("id_field", "id"), row_index))
                    )
                    output_ids = result.decoder_ids
                    cycle_stats.append(result.stats)
                    del source
                _sync(adapter.device)
                elapsed = time.perf_counter() - start
                if adapter.device.type == "cuda":
                    allocated = int(torch.cuda.max_memory_allocated(adapter.device))
                    reserved = int(torch.cuda.max_memory_reserved(adapter.device))
                    if route == "native":
                        native_peak_allocated = max(native_peak_allocated, allocated)
                        native_peak_reserved = max(native_peak_reserved, reserved)
                    else:
                        speculative_peak_allocated = max(speculative_peak_allocated, allocated)
                        speculative_peak_reserved = max(speculative_peak_reserved, reserved)
                outputs[route] = output_ids
                if route == "native":
                    native_times.append(elapsed)
                    native_lengths.append(max(0, len(output_ids) - 1))
                else:
                    speculative_times.append(elapsed)
                    speculative_lengths.append(max(0, len(output_ids) - 1))
            exact_match = outputs["native"] == outputs["speculative"]
            parity.append(exact_match)
            output_diagnostics.append(
                {
                    "repeat": repeat,
                    "native_decoder_ids": outputs["native"],
                    "speculative_decoder_ids": outputs["speculative"],
                }
            )
            if not exact_match:
                mismatch_index = next(
                    (
                        index
                        for index, (native_token, speculative_token) in enumerate(
                            zip(outputs["native"], outputs["speculative"])
                        )
                        if native_token != speculative_token
                    ),
                    min(len(outputs["native"]), len(outputs["speculative"])),
                )
                mismatch_diagnostics.append(
                    {
                        "repeat": repeat,
                        "first_mismatch_position": mismatch_index,
                        "native_length": len(outputs["native"]),
                        "speculative_length": len(outputs["speculative"]),
                        "native_neighborhood": outputs["native"][max(0, mismatch_index - 3) : mismatch_index + 4],
                        "speculative_neighborhood": outputs["speculative"][
                            max(0, mismatch_index - 3) : mismatch_index + 4
                        ],
                    }
                )
        identifier = row.get(config["data"].get("id_field", "id"), row_index)
        request_records.append(
            {
                "id": str(identifier),
                "source_tokens": int(attention_mask[0].sum().item()),
                "native_seconds": _summary(native_times),
                "speculative_seconds": _summary(speculative_times),
                "native_output_tokens": native_lengths,
                "speculative_output_tokens": speculative_lengths,
                "exact_token_match_all_repeats": all(parity),
                "exact_token_match_by_repeat": parity,
                "first_mismatch_by_repeat": mismatch_diagnostics,
                "decoder_ids_by_repeat": output_diagnostics,
                "speedup_median": statistics.median(native_times) / max(statistics.median(speculative_times), 1e-12),
                "speculative_cycles": cycle_stats,
            }
        )

    native_medians = [record["native_seconds"]["median"] for record in request_records]
    speculative_medians = [record["speculative_seconds"]["median"] for record in request_records]
    speedups = [record["speedup_median"] for record in request_records]
    output_root = Path(output_dir or config.get("benchmark", {}).get("output_dir", "./runs/drafter/benchmark"))
    if not output_root.is_absolute():
        output_root = (Path(config["_config_dir"]) / output_root).resolve()
    run_dir = output_root / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for record in request_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "status": "measured",
        "target": adapter.target_manifest(),
        "draft_checkpoint": str(Path(draft_checkpoint).expanduser().resolve()),
        "input_file": str(Path(input_path).expanduser().resolve()),
        "examples": len(request_records),
        "repeats": repeats,
        "warmup_examples": min(warmup, len(rows)),
        "generation_config": config["generation"],
        "source_memory": config["source_memory"],
        "block_size": config["speculation"]["block_size"],
        "native_completion_seconds": _summary(native_medians),
        "speculative_completion_seconds": _summary(speculative_medians),
        "per_request_speedup": _summary(speedups),
        "mean_speedup": statistics.mean(speedups) if speedups else 0.0,
        "exact_token_match_rate": sum(int(record["exact_token_match_all_repeats"]) for record in request_records)
        / max(1, len(request_records)),
        "native_peak_allocated_bytes": native_peak_allocated if adapter.device.type == "cuda" else None,
        "native_peak_reserved_bytes": native_peak_reserved if adapter.device.type == "cuda" else None,
        "speculative_peak_allocated_bytes": speculative_peak_allocated if adapter.device.type == "cuda" else None,
        "speculative_peak_reserved_bytes": speculative_peak_reserved if adapter.device.type == "cuda" else None,
        "cache_note": "Speculative target uses full-history dynamic caches for exact rollback; native generate uses its configured cache.",
    }
    _write_json(run_dir / "manifest.json", {"target": adapter.target_manifest(), "config": config})
    _write_json(run_dir / "summary.json", summary)
    return {"run_dir": str(run_dir), **summary}
