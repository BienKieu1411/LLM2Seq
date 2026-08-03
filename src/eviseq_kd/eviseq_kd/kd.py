"""Knowledge-distillation primitives used by the independent EviSeq-KD run."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def topk_distillation_loss(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Compute masked top-k KL KD with the standard ``T^2`` correction.

    Teacher logits are renormalized over the cached top-k support.  This is
    useful when storing a full teacher distribution is infeasible, but it is
    not enabled by default. The simple EviSeq-KD runner uses sequence KD and
    retokenizes teacher text with the student tokenizer instead.
    """

    if temperature <= 0.0:
        raise ValueError("KD temperature must be positive")
    if student_logits.ndim != 3 or teacher_topk_ids.ndim != 3:
        raise ValueError("Student logits and teacher top-k IDs must be [B,T,*]")
    if teacher_topk_logits.shape != teacher_topk_ids.shape:
        raise ValueError("Teacher top-k IDs and logits must have identical shapes")
    if student_logits.shape[:2] != teacher_topk_ids.shape[:2]:
        raise ValueError("Teacher top-k rows must align with student [B,T] positions")
    if teacher_topk_ids.shape[-1] <= 0:
        raise ValueError("Teacher top-k width must be positive")
    if int(teacher_topk_ids.min().item()) < 0 or int(teacher_topk_ids.max().item()) >= student_logits.shape[-1]:
        raise ValueError("Teacher top-k IDs contain a value outside the student vocabulary")

    ids = teacher_topk_ids.to(device=student_logits.device, dtype=torch.long)
    teacher = teacher_topk_logits.to(device=student_logits.device, dtype=torch.float32)
    selected = student_logits.float().gather(-1, ids)
    scaled_teacher = teacher / float(temperature)
    teacher_probs = F.softmax(scaled_teacher, dim=-1)
    student_log_probs = F.log_softmax(selected / float(temperature), dim=-1)
    per_position = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    per_position = per_position * (float(temperature) ** 2)
    if mask is None:
        valid = torch.ones_like(per_position, dtype=torch.bool)
    else:
        if mask.shape != per_position.shape:
            raise ValueError("KD mask must have shape [B,T]")
        valid = mask.to(device=per_position.device, dtype=torch.bool)
    if not bool(valid.any()):
        return student_logits.float().sum() * 0.0
    return per_position[valid].mean()


def top1_agreement(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Measure how often the student's top-1 token equals the teacher top-1."""

    if student_logits.shape[:2] != teacher_topk_ids.shape[:2] or teacher_topk_ids.shape[-1] <= 0:
        raise ValueError("Teacher top-k IDs must align with student logits")
    student_top1 = student_logits.argmax(dim=-1)
    agreement = student_top1.eq(teacher_topk_ids[..., 0].to(student_top1.device))
    if mask is not None:
        if mask.shape != agreement.shape:
            raise ValueError("KD mask must have shape [B,T]")
        agreement = agreement[mask.to(device=agreement.device, dtype=torch.bool)]
    if agreement.numel() == 0:
        return student_logits.float().sum() * 0.0
    return agreement.float().mean()
