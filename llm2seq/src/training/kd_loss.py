"""
Knowledge Distillation losses for LLM2Seq.

Supports three types of distillation:
1. Sequence KD: Train student on teacher-generated outputs.
2. Logits KL:   Full-vocab KL divergence between teacher and student.
3. Top-k KL:    KL divergence only on teacher's top-k tokens (memory efficient).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_kd_loss(
    student_logits: torch.Tensor,
    teacher_target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Sequence-level distillation: CE loss on teacher-generated outputs.

    Args:
        student_logits: [B, T, V] — student model logits.
        teacher_target_ids: [B, T] — teacher-generated token IDs.
        ignore_index: Token ID to ignore in loss computation.

    Returns:
        Scalar loss.
    """
    loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        teacher_target_ids.view(-1),
        ignore_index=ignore_index,
    )
    return loss


def logits_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Full-vocab KL divergence distillation.

    L_KD = T^2 * KL(softmax(z_T/T) || log_softmax(z_S/T))

    Args:
        student_logits: [B, T, V] — student logits.
        teacher_logits: [B, T, V] — teacher logits (detached).
        temperature: Softmax temperature.
        labels: [B, T] — optional labels to create a valid-position mask.
        ignore_index: Ignore index for masking.

    Returns:
        Scalar loss.
    """
    # Scale by temperature
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # KL divergence: sum over vocab, mean over positions
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

    # Mask invalid positions
    if labels is not None:
        valid_mask = (labels != ignore_index).float()
        kl = kl * valid_mask
        loss = kl.sum() / valid_mask.sum().clamp(min=1.0)
    else:
        loss = kl.mean()

    # Scale by T^2
    loss = loss * (temperature ** 2)
    return loss


def topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    temperature: float = 2.0,
    labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Top-k KL divergence distillation (memory-efficient).

    Only computes KL over the teacher's top-k vocabulary entries.

    Args:
        student_logits: [B, T, V] — full student logits.
        teacher_logits: [B, T, K] — teacher's top-k logits.
        teacher_topk_indices: [B, T, K] — indices of teacher's top-k tokens.
        temperature: Softmax temperature.
        labels: [B, T] — optional labels for masking.
        ignore_index: Ignore index for masking.

    Returns:
        Scalar loss.
    """
    # Gather student logits at teacher's top-k positions
    student_topk = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)

    # Compute KL over top-k subset
    student_log_probs = F.log_softmax(student_topk / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

    # Mask invalid positions
    if labels is not None:
        valid_mask = (labels != ignore_index).float()
        kl = kl * valid_mask
        loss = kl.sum() / valid_mask.sum().clamp(min=1.0)
    else:
        loss = kl.mean()

    # Scale by T^2
    loss = loss * (temperature ** 2)
    return loss


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kd_type: str = "topk_kl",
    temperature: float = 2.0,
    top_k: int = 10000,
    teacher_topk_indices: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Unified KD loss dispatcher.

    Args:
        student_logits: [B, T, V] — student model logits.
        teacher_logits: [B, T, V or K] — teacher logits.
        kd_type: "logits_kl", "topk_kl", or "sequence_kd".
        temperature: Softmax temperature for KL-based methods.
        top_k: Number of top tokens for top-k KL.
        teacher_topk_indices: Pre-computed top-k indices (for cached logits).
        labels: [B, T] — labels for masking.
        ignore_index: Ignore index.

    Returns:
        Scalar KD loss.
    """
    if kd_type == "logits_kl":
        return logits_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits.detach(),
            temperature=temperature,
            labels=labels,
            ignore_index=ignore_index,
        )

    elif kd_type == "topk_kl":
        if teacher_topk_indices is None:
            # Compute top-k on-the-fly from full teacher logits
            teacher_topk_vals, teacher_topk_idx = teacher_logits.detach().topk(
                k=min(top_k, teacher_logits.size(-1)), dim=-1
            )
        else:
            teacher_topk_vals = teacher_logits.detach()
            teacher_topk_idx = teacher_topk_indices

        return topk_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_topk_vals,
            teacher_topk_indices=teacher_topk_idx,
            temperature=temperature,
            labels=labels,
            ignore_index=ignore_index,
        )

    elif kd_type == "sequence_kd":
        # teacher_logits here should be teacher-generated token IDs [B, T]
        return sequence_kd_loss(
            student_logits=student_logits,
            teacher_target_ids=teacher_logits.long(),
            ignore_index=ignore_index,
        )

    else:
        raise ValueError(f"Unknown KD type: {kd_type}. Use 'logits_kl', 'topk_kl', or 'sequence_kd'.")
