"""
Confidence-adaptive accept logic for MTP inference.

Decides how many draft tokens from MTP heads to accept
based on their confidence scores (max probability).
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def confidence_adaptive_accept(
    main_token: torch.Tensor,
    main_confidence: torch.Tensor,
    draft_results: List[dict],
    confidence_threshold: float = 0.9,
) -> Tuple[List[torch.Tensor], int]:
    """
    Accept a prefix of draft tokens based on confidence threshold.

    Accepts tokens sequentially from main + drafts, stopping at the first
    token whose confidence falls below the threshold.

    Args:
        main_token: [B, 1] — main head predicted token.
        main_confidence: [B, 1] — main head confidence.
        draft_results: List of K dicts from MTP heads, each with
            "token_ids" [B, 1] and "confidence" [B, 1].
        confidence_threshold: Minimum confidence to accept a token.

    Returns:
        Tuple of:
            - accepted_tokens: List of accepted token tensors [B, 1].
            - num_accepted: Number of accepted tokens (including main).
    """
    accepted = [main_token]

    for draft in draft_results:
        token = draft["token_ids"]
        conf = draft["confidence"]

        # Check if ALL samples in the batch meet threshold
        # For batch decoding, we use min confidence across the batch
        if conf.min().item() >= confidence_threshold:
            accepted.append(token)
        else:
            break

    return accepted, len(accepted)


def static_accept(
    main_token: torch.Tensor,
    draft_results: List[dict],
    num_accept: int = 2,
) -> Tuple[List[torch.Tensor], int]:
    """
    Accept a fixed number of draft tokens regardless of confidence.

    Args:
        main_token: [B, 1] — main head token.
        draft_results: MTP draft results.
        num_accept: Number of draft tokens to accept.

    Returns:
        (accepted_tokens, num_accepted)
    """
    accepted = [main_token]
    for i, draft in enumerate(draft_results):
        if i >= num_accept:
            break
        accepted.append(draft["token_ids"])

    return accepted, len(accepted)


def compute_acceptance_metrics(
    accepted_lengths: List[int],
    num_mtp_heads: int,
) -> dict:
    """
    Compute MTP acceptance metrics from a list of accepted lengths.

    Args:
        accepted_lengths: List of accepted token counts per step.
        num_mtp_heads: Number of MTP heads.

    Returns:
        Dict with:
            - "acceptance_rate": fraction of draft tokens accepted.
            - "average_accepted_length": mean accepted tokens per step.
            - "cumulative_acceptance_rates": per-depth acceptance rates.
    """
    if not accepted_lengths:
        return {
            "acceptance_rate": 0.0,
            "average_accepted_length": 1.0,
            "cumulative_acceptance_rates": [0.0] * num_mtp_heads,
        }

    total_accepted_drafts = sum(max(0, l - 1) for l in accepted_lengths)
    total_possible_drafts = len(accepted_lengths) * num_mtp_heads

    ar = total_accepted_drafts / max(1, total_possible_drafts)
    avg_len = sum(accepted_lengths) / len(accepted_lengths)

    # Cumulative acceptance rate per depth
    car = []
    for k in range(num_mtp_heads):
        accepted_at_k = sum(1 for l in accepted_lengths if l > k + 1)
        car.append(accepted_at_k / len(accepted_lengths))

    return {
        "acceptance_rate": ar,
        "average_accepted_length": avg_len,
        "cumulative_acceptance_rates": car,
    }
