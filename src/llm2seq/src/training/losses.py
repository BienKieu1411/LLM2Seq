"""
Total loss computation for LLM2Seq.

Combines CE loss, optional KD loss, and optional MTP loss:
    L_total = L_CE + λ_KD * I_distill * L_KD + λ_MTP * I_mtp * L_MTP
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .kd_loss import compute_kd_loss
from .mtp_loss import compute_mtp_loss, compute_mtp_self_distillation_loss


def compute_total_loss(
    main_logits: torch.Tensor,
    labels: torch.Tensor,
    mtp_logits: Optional[List[torch.Tensor]] = None,
    teacher_logits: Optional[torch.Tensor] = None,
    teacher_topk_indices: Optional[torch.Tensor] = None,
    cfg: Any = None,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Compute the total training loss with optional components.

    Args:
        main_logits: [B, T, V] — main decoder head logits.
        labels: [B, T] — target token IDs.
        mtp_logits: List of MTP head logits (if MTP enabled).
        teacher_logits: Teacher logits for distillation.
        teacher_topk_indices: Teacher top-k indices (for cached logits).
        cfg: Model configuration with feature flags and hyperparameters.
        ignore_index: Ignore index for CE loss.

    Returns:
        Dict with "loss" (total), "loss_ce", "loss_kd", "loss_mtp".
    """
    # 1. Main cross-entropy loss
    loss_ce = F.cross_entropy(
        main_logits.view(-1, main_logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )

    total_loss = loss_ce.clone()
    result: Dict[str, torch.Tensor] = {"loss_ce": loss_ce}

    # 2. Knowledge Distillation loss (if enabled)
    loss_kd = torch.tensor(0.0, device=labels.device)
    if cfg is not None and getattr(cfg, "use_distillation", False) and teacher_logits is not None:
        loss_kd = compute_kd_loss(
            student_logits=main_logits,
            teacher_logits=teacher_logits,
            kd_type=getattr(cfg, "kd_type", "topk_kl"),
            temperature=getattr(cfg, "kd_temperature", 2.0),
            top_k=getattr(cfg, "kd_top_k", 10000),
            teacher_topk_indices=teacher_topk_indices,
            labels=labels,
            ignore_index=ignore_index,
        )
        kd_weight = getattr(cfg, "kd_loss_weight", 0.5)
        total_loss = total_loss + kd_weight * loss_kd

    result["loss_kd"] = loss_kd

    # 3. MTP loss (if enabled)
    loss_mtp = torch.tensor(0.0, device=labels.device)
    loss_mtp_ce = torch.tensor(0.0, device=labels.device)
    loss_mtp_kl = torch.tensor(0.0, device=labels.device)
    if cfg is not None and getattr(cfg, "use_mtp", False) and mtp_logits is not None:
        loss_mtp_ce = compute_mtp_loss(
            mtp_logits=mtp_logits,
            labels=labels,
            head_weights=getattr(cfg, "mtp_head_weights", None),
            ignore_index=ignore_index,
        )
        mtp_weight = getattr(cfg, "mtp_loss_weight", 0.3)
        loss_mtp = loss_mtp_ce

        if getattr(cfg, "mtp_self_distillation", False):
            loss_mtp_kl = compute_mtp_self_distillation_loss(
                mtp_logits=mtp_logits,
                main_logits=main_logits,
                labels=labels,
                top_k=getattr(cfg, "mtp_self_distill_top_k", 10000),
                head_weights=getattr(cfg, "mtp_self_distill_head_weights", None),
                temperature=getattr(cfg, "mtp_self_distill_temperature", 1.0),
                ignore_index=ignore_index,
            )

        mtp_kl_weight = getattr(cfg, "mtp_self_distill_loss_weight", 0.5)
        mtp_total = mtp_weight * loss_mtp_ce + mtp_kl_weight * loss_mtp_kl
        if getattr(cfg, "mtp_train_only", False):
            total_loss = mtp_total
        else:
            total_loss = total_loss + mtp_total

    result["loss_mtp"] = loss_mtp
    result["loss_mtp_ce"] = loss_mtp_ce
    result["loss_mtp_kl"] = loss_mtp_kl
    result["loss"] = total_loss

    return result
