import json
import statistics
import math
from pathlib import Path

# Fix pythonpath
import sys
sys.path.append(".")

from llm2seq.scripts.evaluate_full_test import compute_metrics, aggregate_mtp_metrics, safe_mean, safe_median, percentile

def recover():
    predictions_path = Path("runs/phase3_mtp_self_distill_qwen_wikilingua/eval_mtp/predictions.jsonl")
    metrics_path = Path("runs/phase3_mtp_self_distill_qwen_wikilingua/eval_mtp/metrics.json")
    
    if not predictions_path.exists():
        print("predictions.jsonl not found!")
        return

    predictions = []
    references = []
    sources = []
    latencies = []
    new_token_counts = []
    decode_steps = []
    mtp_metrics_list = []
    
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            predictions.append(row["prediction"])
            references.append(row["reference"])
            sources.append(row["source"])
            latencies.append(row["latency_seconds"])
            new_token_counts.append(row["generated_tokens"])
            decode_steps.append(row["decode_steps"])
            if "mtp_metrics" in row:
                mtp_metrics_list.append(row["mtp_metrics"])

    print(f"Loaded {len(predictions)} predictions.")
    
    metrics = compute_metrics(
        predictions,
        references,
        sources=sources,
        compute_bertscore=False,
    )
    
    total_new_tokens = sum(new_token_counts)
    
    metrics.update(
        {
            "decode_mode": "mtp_verified",
            "decode_elapsed_seconds": round(sum(latencies), 3),
            "decode_examples_per_second": round(len(predictions) / max(sum(latencies), 1e-9), 6),
            "decode_generated_tokens_per_second": round(total_new_tokens / max(sum(latencies), 1e-9), 3),
            "seconds_per_generated_token": round(sum(latencies) / max(total_new_tokens, 1), 8),
            "latency_seconds_mean": round(safe_mean(latencies), 6),
            "latency_seconds_median": round(safe_median(latencies), 6),
            "latency_seconds_p95": round(percentile(latencies, 95), 6),
            "latency_seconds_min": round(min(latencies), 6) if latencies else 0.0,
            "latency_seconds_max": round(max(latencies), 6) if latencies else 0.0,
            "generated_tokens_total": total_new_tokens,
            "generated_tokens_mean": round(safe_mean(new_token_counts), 4),
            "decode_steps_total": round(sum(decode_steps), 6),
            "decode_steps_mean": round(safe_mean(decode_steps), 6),
            "tokens_per_decode_step": round(total_new_tokens / max(sum(decode_steps), 1e-9), 6),
        }
    )
    
    metrics.update(aggregate_mtp_metrics(mtp_metrics_list))
    
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        
    print(f"Recovered and saved metrics to {metrics_path}")

if __name__ == "__main__":
    recover()
