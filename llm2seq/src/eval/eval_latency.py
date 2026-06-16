"""
Latency and throughput evaluation for LLM2Seq.

Measures:
- Average latency per sample
- Tokens per second
- Peak GPU memory usage
- KV-cache memory estimate
- Average decoding steps
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch


def evaluate_latency(
    model,
    dataloader,
    tokenizer,
    generate_fn,
    device: torch.device,
    max_new_tokens: int = 128,
    num_warmup: int = 5,
    num_samples: int = 50,
    use_mtp: bool = False,
) -> Dict[str, float]:
    """
    Benchmark inference latency and throughput.

    Args:
        model: LLM2Seq model.
        dataloader: DataLoader providing batches.
        tokenizer: Tokenizer for decoding.
        generate_fn: Generation function (autoregressive_generate or mtp_generate).
        device: Device for computation.
        max_new_tokens: Max tokens to generate per sample.
        num_warmup: Number of warmup iterations (not timed).
        num_samples: Number of samples to benchmark.
        use_mtp: Whether MTP generation is being used.

    Returns:
        Dict with latency, throughput, and memory metrics.
    """
    model.eval()

    latencies: List[float] = []
    total_tokens = 0
    total_steps = 0
    sample_count = 0

    # Reset peak memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= num_warmup + num_samples:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Generate
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.perf_counter()

        if use_mtp:
            result = generate_fn(
                model=model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )
            generated_ids = result["generated_ids"]
            steps = result.get("num_steps", generated_ids.size(1))
        else:
            generated_ids = generate_fn(
                model=model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )
            steps = generated_ids.size(1)

        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start_time

        # Skip warmup
        if batch_idx >= num_warmup:
            bsz = generated_ids.size(0)
            num_tokens = generated_ids.size(1) * bsz
            latencies.append(elapsed / bsz)
            total_tokens += num_tokens
            total_steps += steps * bsz
            sample_count += bsz

    # Compute metrics
    avg_latency = sum(latencies) / max(1, len(latencies))
    total_time = sum(latencies) * (total_tokens / max(1, total_tokens))  # approximate
    tokens_per_second = total_tokens / max(1e-6, sum(latencies) * len(latencies) / max(1, len(latencies)))

    result = {
        "avg_latency_per_sample_sec": round(avg_latency, 4),
        "tokens_per_second": round(total_tokens / max(1e-6, sum(latencies)), 2),
        "total_samples": sample_count,
        "total_tokens_generated": total_tokens,
        "avg_decoding_steps": round(total_steps / max(1, sample_count), 2),
    }

    # GPU memory
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        result["peak_gpu_memory_mb"] = round(peak_memory_mb, 2)

    return result
