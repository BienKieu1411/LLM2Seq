"""
MTP acceptance rate evaluation for LLM2Seq.

Measures:
- Acceptance Rate (AR): fraction of MTP tokens accepted.
- Cumulative Acceptance Rate (CAR): acceptance rate at each depth.
- Average accepted length per decoding step.
- Speedup ratio vs autoregressive.
"""

from __future__ import annotations

from typing import Dict, List

import torch


def evaluate_acceptance_rate(
    model,
    dataloader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 128,
    confidence_threshold: float = 0.9,
    num_samples: int = 100,
) -> Dict[str, float]:
    """
    Evaluate MTP acceptance rate metrics.

    Args:
        model: LLM2Seq model with MTP module.
        dataloader: DataLoader providing batches.
        tokenizer: Tokenizer for decoding.
        device: Computation device.
        max_new_tokens: Max tokens per sample.
        confidence_threshold: Confidence threshold for acceptance.
        num_samples: Number of samples to evaluate.

    Returns:
        Dict with acceptance rate metrics.
    """
    from ..inference.generate_mtp import mtp_generate

    model.eval()

    all_metrics: List[dict] = []
    sample_count = 0

    for batch in dataloader:
        if sample_count >= num_samples:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        result = mtp_generate(
            model=model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            confidence_threshold=confidence_threshold,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )

        all_metrics.append(result["metrics"])
        sample_count += batch["input_ids"].size(0)

    # Aggregate metrics
    if not all_metrics:
        return {
            "acceptance_rate": 0.0,
            "average_accepted_length": 1.0,
            "speedup_vs_autoregressive": 1.0,
            "avg_num_steps": 0,
        }

    avg_ar = sum(m["acceptance_rate"] for m in all_metrics) / len(all_metrics)
    avg_aal = sum(m["average_accepted_length"] for m in all_metrics) / len(all_metrics)
    avg_speedup = sum(m["speedup_vs_autoregressive"] for m in all_metrics) / len(all_metrics)
    avg_steps = sum(m["num_steps"] for m in all_metrics) / len(all_metrics)

    # Cumulative acceptance rates per depth
    num_heads = len(all_metrics[0].get("cumulative_acceptance_rates", []))
    avg_car = []
    for k in range(num_heads):
        car_k = sum(m["cumulative_acceptance_rates"][k] for m in all_metrics if k < len(m["cumulative_acceptance_rates"]))
        avg_car.append(car_k / len(all_metrics))

    return {
        "acceptance_rate": round(avg_ar, 4),
        "average_accepted_length": round(avg_aal, 4),
        "speedup_vs_autoregressive": round(avg_speedup, 4),
        "avg_num_steps": round(avg_steps, 2),
        "cumulative_acceptance_rates": [round(c, 4) for c in avg_car],
    }
