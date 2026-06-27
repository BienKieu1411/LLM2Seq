"""LLM2Seq training utilities."""

from .losses import compute_total_loss
from .kd_loss import compute_kd_loss
from .mtp_loss import compute_mtp_loss

__all__ = [
    "compute_total_loss",
    "compute_kd_loss",
    "compute_mtp_loss",
]
