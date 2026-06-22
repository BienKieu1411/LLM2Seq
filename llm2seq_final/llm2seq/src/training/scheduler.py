"""
Learning rate scheduler utilities for LLM2Seq.

Supports:
- Cosine annealing with warmup.
- Per-component learning rates (encoder, adaptor, decoder).
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create a cosine annealing schedule with linear warmup.

    Args:
        optimizer: The optimizer.
        num_warmup_steps: Steps for linear warmup.
        num_training_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of peak LR.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    encoder_lr: float = 1e-5,
    adaptor_lr: float = 2e-4,
    decoder_lr: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    max_steps: int = 50000,
    min_lr_ratio: float = 0.1,
) -> tuple:
    """
    Build optimizer with per-component learning rates and cosine scheduler.

    Parameter groups:
    1. Encoder parameters (lowest LR, possibly frozen).
    2. Adaptor parameters.
    3. Decoder + LM head + MTP parameters.

    Args:
        model: The LLM2Seq model.
        encoder_lr: Learning rate for encoder.
        adaptor_lr: Learning rate for adaptor.
        decoder_lr: Learning rate for decoder + LM head + MTP.
        weight_decay: Weight decay coefficient.
        warmup_steps: Number of warmup steps.
        max_steps: Total number of training steps.
        min_lr_ratio: Minimum LR ratio for cosine schedule.

    Returns:
        (optimizer, scheduler) tuple.
    """
    # Classify parameters into groups
    encoder_params = []
    adaptor_params = []
    decoder_params = []

    no_decay_keywords = {"bias", "layernorm", "layer_norm", "rmsnorm"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine which component this parameter belongs to
        if name.startswith("encoder."):
            group = encoder_params
            lr = encoder_lr
        elif name.startswith("adaptor."):
            group = adaptor_params
            lr = adaptor_lr
        else:
            # decoder, lm_head, mtp_module
            group = decoder_params
            lr = decoder_lr

        # Check if this param should have weight decay
        apply_wd = not any(kw in name.lower() for kw in no_decay_keywords)

        group.append({
            "params": [param],
            "lr": lr,
            "weight_decay": weight_decay if apply_wd else 0.0,
        })

    # Flatten into param groups
    param_groups = encoder_params + adaptor_params + decoder_params

    if not param_groups:
        raise ValueError("No trainable parameters found!")

    optimizer = AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        min_lr_ratio=min_lr_ratio,
    )

    return optimizer, scheduler
