"""
Multi-Token Prediction (MTP) loss for LLM2Seq.

Computes weighted cross-entropy loss across multiple MTP heads,
where each head predicts a future token at a different offset.

L_MTP = Σ_k α_k * CE(mtp_logits[k], y_{t+k})
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def compute_mtp_loss(
    mtp_logits: List[torch.Tensor],
    labels: torch.Tensor,
    head_weights: Optional[List[float]] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute weighted MTP loss across all prediction heads.

    Each MTP head k predicts the token at position t+k+1, so we shift
    the labels accordingly for each head.

    Args:
        mtp_logits: List of K tensors, each [B, T, V].
            mtp_logits[k] predicts y_{t+k+1}.
        labels: [B, T] — ground truth target token IDs.
        head_weights: List of K weights α_k for each head.
            Defaults to uniform weights if not provided.
        ignore_index: Token ID to ignore in CE computation.

    Returns:
        Scalar weighted MTP loss.
    """
    num_heads = len(mtp_logits)

    if head_weights is None:
        head_weights = [1.0] * num_heads
    else:
        # Ensure we have enough weights
        if len(head_weights) < num_heads:
            head_weights = head_weights + [head_weights[-1]] * (num_heads - len(head_weights))

    total_loss = torch.tensor(0.0, device=labels.device, dtype=mtp_logits[0].dtype)
    total_weight = 0.0

    for k in range(num_heads):
        # MTP head k predicts token at offset (k+1) from the current position
        shift = k + 1
        seq_len = labels.size(1)

        if shift >= seq_len:
            # No valid positions to compute loss for this head
            continue

        # Shifted labels: position t should predict label at t+shift
        # mtp_logits[k] at position t predicts y_{t+shift}
        shifted_labels = labels[:, shift:].contiguous()  # [B, T-shift]
        logits_k = mtp_logits[k][:, : seq_len - shift, :].contiguous()  # [B, T-shift, V]

        loss_k = F.cross_entropy(
            logits_k.view(-1, logits_k.size(-1)),
            shifted_labels.view(-1),
            ignore_index=ignore_index,
        )

        total_loss = total_loss + head_weights[k] * loss_k
        total_weight += head_weights[k]

    # Normalize by total weight
    if total_weight > 0:
        total_loss = total_loss / total_weight

    return total_loss


def compute_mtp_self_distillation_loss(
    mtp_logits: List[torch.Tensor],
    main_logits: torch.Tensor,
    labels: torch.Tensor,
    top_k: int = 10000,
    head_weights: Optional[List[float]] = None,
    temperature: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Gradient-detached, TopK-selected forward-KL self-distillation for MTP-D.

    The main head is the teacher. Its logits are detached, top-k vocabulary
    indices are selected per future position, and each MTP head is trained to
    match the main-head distribution on those selected indices.
    """
    num_heads = len(mtp_logits)
    if num_heads == 0:
        return torch.tensor(0.0, device=labels.device)

    if head_weights is None:
        head_weights = [1.0] * num_heads
    elif len(head_weights) < num_heads:
        head_weights = head_weights + [head_weights[-1]] * (num_heads - len(head_weights))

    seq_len = labels.size(1)
    vocab_size = main_logits.size(-1)
    top_k = min(int(top_k), vocab_size)
    temperature = max(float(temperature), 1e-6)

    total_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
    total_weight = 0.0

    for k, logits in enumerate(mtp_logits):
        shift = k + 1
        if shift >= seq_len:
            continue

        student_logits = logits[:, : seq_len - shift, :].contiguous()
        teacher_logits = main_logits[:, shift:, :].detach().contiguous()
        shifted_labels = labels[:, shift:].contiguous()
        valid_mask = shifted_labels.ne(ignore_index)

        if not valid_mask.any():
            continue

        teacher_selected, top_indices = torch.topk(teacher_logits.float(), k=top_k, dim=-1)
        student_selected = torch.gather(student_logits.float(), dim=-1, index=top_indices)

        teacher_probs = F.softmax(teacher_selected / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_selected / temperature, dim=-1)
        kl_per_token = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)
        kl = kl_per_token.masked_select(valid_mask).mean() * (temperature**2)

        total_loss = total_loss + head_weights[k] * kl
        total_weight += head_weights[k]

    if total_weight > 0:
        total_loss = total_loss / total_weight

    return total_loss
