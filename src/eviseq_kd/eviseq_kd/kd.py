"""Knowledge-distillation losses used by the independent EviSeq-KD run."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _validate_logits(logits: torch.Tensor, name: str) -> None:
    if logits.ndim != 3:
        raise ValueError(f"{name} must be [B,T,V], got shape {tuple(logits.shape)}")
    if logits.shape[-1] <= 0:
        raise ValueError(f"{name} must have a positive vocabulary width")


def _valid_positions(
    shape: torch.Size,
    *,
    mask: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Combine an explicit KD mask with the usual ``labels != -100`` mask."""

    valid = torch.ones(shape, dtype=torch.bool, device=device)
    if labels is not None:
        if labels.shape != shape:
            raise ValueError(f"KD labels must have shape [B,T], got {tuple(labels.shape)}")
        valid &= labels.to(device=device).ne(-100)
    if mask is not None:
        if mask.shape != shape:
            raise ValueError(f"KD mask must have shape [B,T], got {tuple(mask.shape)}")
        valid &= mask.to(device=device, dtype=torch.bool)
    return valid


def _masked_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    zero_reference: torch.Tensor,
) -> torch.Tensor:
    if not bool(valid.any()):
        # Keep the result connected to the student graph while avoiding the
        # NaN produced by a mean over an all-padding tensor.
        return zero_reference.float().sum() * 0.0
    return values[valid].mean()


def sequence_kd_loss(
    student_logits: torch.Tensor,
    teacher_target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy on teacher-generated token IDs."""

    _validate_logits(student_logits, "student_logits")
    if teacher_target_ids.ndim != 2 or teacher_target_ids.shape != student_logits.shape[:2]:
        raise ValueError("teacher_target_ids must match student logits in [B,T] shape")
    valid = teacher_target_ids.to(device=student_logits.device).ne(ignore_index)
    if not bool(valid.any()):
        return student_logits.float().sum() * 0.0
    target = teacher_target_ids.to(device=student_logits.device, dtype=torch.long)
    selected_target = target[valid]
    if bool(selected_target.lt(0).any()) or bool(selected_target.ge(student_logits.shape[-1]).any()):
        raise ValueError("Teacher sequence target contains an ID outside the student vocabulary")
    return F.cross_entropy(student_logits.float()[valid], selected_target)


def logits_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    labels: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Full-vocabulary ``T^2 * KL(teacher || student)``.

    ``teacher_logits`` are detached before constructing probabilities.  The
    optional labels and mask are intersected, so decoder padding can never
    contribute to the loss even when a caller supplies a too-broad mask.
    """

    _validate_logits(student_logits, "student_logits")
    _validate_logits(teacher_logits, "teacher_logits")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "Full-vocabulary KL requires teacher_logits and student_logits to have identical [B,T,V] shapes"
        )
    if temperature <= 0.0:
        raise ValueError("KD temperature must be positive")
    if labels is not None and labels.shape != student_logits.shape[:2]:
        raise ValueError("KD labels must have shape [B,T]")
    if labels is not None and ignore_index != -100:
        labels = labels.masked_fill(labels.eq(ignore_index), -100)
    valid = _valid_positions(
        student_logits.shape[:2],
        mask=mask,
        labels=labels,
        device=student_logits.device,
    )
    student_log_probs = F.log_softmax(student_logits.float() / float(temperature), dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach().float() / float(temperature), dim=-1)
    per_position = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return _masked_mean(per_position, valid, student_logits) * (float(temperature) ** 2)


def _validate_topk_ids(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
) -> torch.Tensor:
    if teacher_topk_ids.ndim != 3:
        raise ValueError(f"teacher_topk_ids must be [B,T,K], got shape {tuple(teacher_topk_ids.shape)}")
    if teacher_topk_ids.shape[:2] != student_logits.shape[:2]:
        raise ValueError("Teacher top-k rows must align with student [B,T] positions")
    width = int(teacher_topk_ids.shape[-1])
    vocab = int(student_logits.shape[-1])
    if width <= 0:
        raise ValueError("Teacher top-k width must be positive")
    if width > vocab:
        raise ValueError(f"Teacher top-k width K={width} cannot exceed student vocabulary V={vocab}")
    ids = teacher_topk_ids.to(device=student_logits.device, dtype=torch.long)
    if ids.numel() and (bool(ids.lt(0).any()) or bool(ids.ge(vocab).any())):
        raise ValueError("Teacher top-k IDs are not aligned to the student vocabulary")
    if width > 1:
        sorted_ids = ids.sort(dim=-1).values
        if bool(sorted_ids[..., 1:].eq(sorted_ids[..., :-1]).any()):
            raise ValueError("Teacher top-k IDs contain duplicate token IDs in a row")
    return ids


def topk_distillation_loss(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 2.0,
    labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    teacher_log_normalizers: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute masked top-k KD with the standard ``T^2`` correction.

    Current caches provide the teacher's full-vocabulary log normalizer at the
    configured temperature. In that case this computes an exact KL after
    collapsing every non-top-k token into one ``OTHER`` bucket. This preserves
    the teacher/student probability mass outside the cached support instead of
    renormalizing the student over only K tokens.

    The no-normalizer branch is retained only for direct primitive use and
    legacy tests; production logit KD requires a version-3 cache.
    """

    _validate_logits(student_logits, "student_logits")
    if teacher_topk_logits.shape != teacher_topk_ids.shape:
        raise ValueError("Teacher top-k IDs and logits must have identical [B,T,K] shapes")
    if temperature <= 0.0:
        raise ValueError("KD temperature must be positive")
    ids = _validate_topk_ids(student_logits, teacher_topk_ids)
    if labels is not None and labels.shape != student_logits.shape[:2]:
        raise ValueError("KD labels must have shape [B,T]")
    if labels is not None and ignore_index != -100:
        labels = labels.masked_fill(labels.eq(ignore_index), -100)

    teacher = teacher_topk_logits.detach().to(device=student_logits.device, dtype=torch.float32)
    if teacher_log_normalizers is None:
        selected_student = student_logits.float().gather(-1, ids)
        student_log_probs = F.log_softmax(selected_student / float(temperature), dim=-1)
        teacher_probs = F.softmax(teacher / float(temperature), dim=-1)
        per_position = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    else:
        if teacher_log_normalizers.shape != student_logits.shape[:2]:
            raise ValueError("Teacher log normalizers must have shape [B,T]")
        normalizers = teacher_log_normalizers.detach().to(
            device=student_logits.device,
            dtype=torch.float32,
        )
        if not bool(torch.isfinite(normalizers).all()):
            raise ValueError("Teacher log normalizers must be finite")

        student_full_log_probs = F.log_softmax(student_logits.float() / float(temperature), dim=-1)
        student_topk_log_probs = student_full_log_probs.gather(-1, ids)
        teacher_topk_log_probs = teacher / float(temperature) - normalizers.unsqueeze(-1)
        teacher_topk_probs = teacher_topk_log_probs.exp()
        # Cached logits may be float16 while the normalizer is float32. Guard
        # against a tiny rounding overshoot above unit probability mass.
        teacher_topk_probs = (
            teacher_topk_probs
            / teacher_topk_probs.sum(
                dim=-1,
                keepdim=True,
            )
            .clamp_min(1.0)
            .detach()
        )
        teacher_topk_log_probs = teacher_topk_probs.clamp_min(1.0e-30).log()
        teacher_tail = (1.0 - teacher_topk_probs.sum(dim=-1)).clamp(min=0.0, max=1.0)
        student_tail = (1.0 - student_topk_log_probs.exp().sum(dim=-1)).clamp_min(1.0e-12)

        topk_kl = (teacher_topk_probs * (teacher_topk_log_probs - student_topk_log_probs)).sum(dim=-1)
        tail_kl = torch.where(
            teacher_tail > 0.0,
            teacher_tail * (teacher_tail.clamp_min(1.0e-12).log() - student_tail.log()),
            torch.zeros_like(teacher_tail),
        )
        per_position = topk_kl + tail_kl
    valid = _valid_positions(
        student_logits.shape[:2],
        mask=mask,
        labels=labels,
        device=student_logits.device,
    )
    return _masked_mean(per_position, valid, student_logits) * (float(temperature) ** 2)


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kd_type: str = "topk_kl",
    temperature: float = 2.0,
    top_k: int = 10000,
    teacher_topk_indices: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Dispatch sequence, full-vocabulary, or cached top-k KD."""

    if kd_type == "sequence_kd":
        return sequence_kd_loss(student_logits, teacher_logits.long(), ignore_index=ignore_index)
    if kd_type == "logits_kl":
        return logits_kl_loss(
            student_logits,
            teacher_logits,
            temperature=temperature,
            labels=labels,
            mask=mask,
            ignore_index=ignore_index,
        )
    if kd_type == "topk_kl":
        if teacher_topk_indices is None:
            _validate_logits(teacher_logits, "teacher_logits")
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            width = min(int(top_k), int(teacher_logits.shape[-1]))
            teacher_values, teacher_indices = teacher_logits.detach().topk(width, dim=-1)
        else:
            teacher_values = teacher_logits
            teacher_indices = teacher_topk_indices
        return topk_distillation_loss(
            student_logits,
            teacher_indices,
            teacher_values,
            mask=mask,
            temperature=temperature,
            labels=labels,
            ignore_index=ignore_index,
        )
    raise ValueError("Unknown KD type: use 'logits_kl', 'topk_kl', or 'sequence_kd'")


def top1_agreement(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Measure how often the student's top-1 token equals teacher top-1."""

    _validate_logits(student_logits, "student_logits")
    ids = _validate_topk_ids(student_logits, teacher_topk_ids)
    agreement = student_logits.argmax(dim=-1).eq(ids[..., 0])
    if mask is not None:
        if mask.shape != agreement.shape:
            raise ValueError("KD mask must have shape [B,T]")
        agreement = agreement[mask.to(device=agreement.device, dtype=torch.bool)]
    if agreement.numel() == 0:
        return student_logits.float().sum() * 0.0
    return agreement.float().mean()
