#!/usr/bin/env python3
"""Compare main-head and D-MTP evaluation speed metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def percent_change(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return 0.0
    return 100.0 * (new_value - old_value) / old_value


def get_float(metrics: Dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    return float(value) if value is not None else 0.0


def metric_delta(main_metrics: Dict[str, Any], mtp_metrics: Dict[str, Any], key: str) -> Dict[str, float]:
    main_value = get_float(main_metrics, key)
    mtp_value = get_float(mtp_metrics, key)
    return {
        "main": round(main_value, 6),
        "mtp_verified": round(mtp_value, 6),
        "delta": round(mtp_value - main_value, 6),
    }


def build_comparison(main_metrics: Dict[str, Any], mtp_metrics: Dict[str, Any]) -> Dict[str, Any]:
    main_decode_tps = get_float(main_metrics, "decode_generated_tokens_per_second")
    mtp_decode_tps = get_float(mtp_metrics, "decode_generated_tokens_per_second")
    main_wall_tps = get_float(main_metrics, "generated_tokens_per_second")
    mtp_wall_tps = get_float(mtp_metrics, "generated_tokens_per_second")
    main_decode_eps = get_float(main_metrics, "decode_examples_per_second")
    mtp_decode_eps = get_float(mtp_metrics, "decode_examples_per_second")
    main_latency_mean = get_float(main_metrics, "latency_seconds_mean")
    mtp_latency_mean = get_float(mtp_metrics, "latency_seconds_mean")
    main_latency_p95 = get_float(main_metrics, "latency_seconds_p95")
    mtp_latency_p95 = get_float(mtp_metrics, "latency_seconds_p95")
    main_step_tokens = get_float(main_metrics, "tokens_per_decode_step")
    mtp_step_tokens = get_float(mtp_metrics, "tokens_per_decode_step")

    return {
        "summary": {
            "real_decode_tokens_per_second_speedup": round(ratio(mtp_decode_tps, main_decode_tps), 6),
            "real_decode_examples_per_second_speedup": round(ratio(mtp_decode_eps, main_decode_eps), 6),
            "wall_clock_tokens_per_second_speedup": round(ratio(mtp_wall_tps, main_wall_tps), 6),
            "latency_mean_speedup": round(ratio(main_latency_mean, mtp_latency_mean), 6),
            "latency_p95_speedup": round(ratio(main_latency_p95, mtp_latency_p95), 6),
            "latency_mean_change_percent": round(percent_change(mtp_latency_mean, main_latency_mean), 4),
            "latency_p95_change_percent": round(percent_change(mtp_latency_p95, main_latency_p95), 4),
            "tokens_per_decode_step_gain": round(ratio(mtp_step_tokens, main_step_tokens), 6),
            "mtp_theoretical_decode_step_speedup": round(
                get_float(mtp_metrics, "mtp_speedup_vs_autoregressive_mean"), 6
            ),
            "mtp_acceptance_rate_mean": round(get_float(mtp_metrics, "mtp_acceptance_rate_mean"), 6),
            "mtp_average_accepted_length_mean": round(get_float(mtp_metrics, "mtp_average_accepted_length_mean"), 6),
            "mtp_average_emitted_length_mean": round(get_float(mtp_metrics, "mtp_average_emitted_length_mean"), 6),
            "mtp_fallback_to_autoregressive_rate": round(
                get_float(mtp_metrics, "mtp_fallback_to_autoregressive_rate"), 6
            ),
        },
        "main": {
            "decode_mode": main_metrics.get("decode_mode"),
            "decode_generated_tokens_per_second": round(main_decode_tps, 6),
            "decode_examples_per_second": round(main_decode_eps, 6),
            "generated_tokens_per_second": round(main_wall_tps, 6),
            "latency_seconds_mean": round(main_latency_mean, 6),
            "latency_seconds_p95": round(main_latency_p95, 6),
            "tokens_per_decode_step": round(main_step_tokens, 6),
            "peak_gpu_memory_mb": round(get_float(main_metrics, "peak_gpu_memory_mb"), 3),
        },
        "mtp_verified": {
            "decode_mode": mtp_metrics.get("decode_mode"),
            "decode_generated_tokens_per_second": round(mtp_decode_tps, 6),
            "decode_examples_per_second": round(mtp_decode_eps, 6),
            "generated_tokens_per_second": round(mtp_wall_tps, 6),
            "latency_seconds_mean": round(mtp_latency_mean, 6),
            "latency_seconds_p95": round(mtp_latency_p95, 6),
            "tokens_per_decode_step": round(mtp_step_tokens, 6),
            "peak_gpu_memory_mb": round(get_float(mtp_metrics, "peak_gpu_memory_mb"), 3),
            "fallback_to_autoregressive_rate": round(get_float(mtp_metrics, "mtp_fallback_to_autoregressive_rate"), 6),
        },
        "quality_delta": {
            "rouge1": metric_delta(main_metrics, mtp_metrics, "rouge1"),
            "rouge2": metric_delta(main_metrics, mtp_metrics, "rouge2"),
            "rougeL": metric_delta(main_metrics, mtp_metrics, "rougeL"),
            "rougeLsum": metric_delta(main_metrics, mtp_metrics, "rougeLsum"),
            "bleu": metric_delta(main_metrics, mtp_metrics, "bleu"),
            "chrf": metric_delta(main_metrics, mtp_metrics, "chrf"),
        },
        "files": {
            "main_metrics": str(main_metrics.get("metrics_file", "")),
            "mtp_metrics": str(mtp_metrics.get("metrics_file", "")),
        },
        "note": (
            "Use real_decode_tokens_per_second_speedup and latency_*_speedup as the real measured speedup. "
            "mtp_theoretical_decode_step_speedup only measures fewer verifier steps, not full serving latency."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_metrics", required=True)
    parser.add_argument("--mtp_metrics", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    main_metrics_path = Path(args.main_metrics)
    mtp_metrics_path = Path(args.mtp_metrics)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_metrics = load_json(main_metrics_path)
    mtp_metrics = load_json(mtp_metrics_path)
    main_metrics["metrics_file"] = str(main_metrics_path)
    mtp_metrics["metrics_file"] = str(mtp_metrics_path)

    comparison = build_comparison(main_metrics, mtp_metrics)
    output_path = output_dir / "speed_comparison.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(json.dumps(comparison["summary"], ensure_ascii=False, indent=2))
    print(f"Saved speed comparison to {output_path}")


if __name__ == "__main__":
    main()
