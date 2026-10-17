"""Small mathematical reference for checks; production uses TRL's objective."""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, group_size: int, scale: str = "none") -> torch.Tensor:
    if group_size < 2 or rewards.ndim != 1 or rewards.numel() % group_size:
        raise ValueError("Each local prompt must have a complete group of >=2 rewards")
    if not torch.isfinite(rewards).all():
        raise ValueError("Non-finite rewards")
    grouped = rewards.detach().float().reshape(-1, group_size)
    centered = grouped - grouped.mean(-1, keepdim=True)
    if scale == "group":
        centered = centered / grouped.std(-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    elif scale != "none":
        raise ValueError("scale_rewards must be none or group")
    return centered.flatten()


def clipped_terms(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float = 0.2,
    ref_logps: torch.Tensor | None = None,
    beta: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return SUM of token losses, leaving global normalization to the caller.

    Mask before exponentiation: padded values cannot inject NaNs/overflow.
    Never detach the current-policy term, including at the first on-policy step.
    """
    if current_logps.shape != old_logps.shape or current_logps.shape != mask.shape:
        raise ValueError("Current/behavior log-probs and action mask must have equal shapes")
    if advantages.shape != current_logps.shape[:1] or not 0 < epsilon < 1:
        raise ValueError("Invalid sequence advantages or clipping epsilon")
    active = mask.bool()
    delta = (current_logps - old_logps.detach()).masked_fill(~active, 0)
    if not torch.isfinite(delta).all():
        raise FloatingPointError("Non-finite policy log-ratio")
    ratio = delta.exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("Policy ratio overflow; reduce learning rate / inspect log-probs")
    adv = advantages.detach()[:, None]
    policy = -torch.minimum(ratio * adv, ratio.clamp(1 - epsilon, 1 + epsilon) * adv)
    kl = torch.zeros_like(policy)
    if beta:
        if ref_logps is None or ref_logps.shape != current_logps.shape:
            raise ValueError("beta > 0 requires frozen SFT reference log-probs")
        ref_delta = (ref_logps.detach() - current_logps).masked_fill(~active, 0)
        kl = ref_delta.exp() - ref_delta - 1
        if not torch.isfinite(kl).all():
            raise FloatingPointError("Non-finite reference KL estimator")
    terms = (policy + beta * kl).masked_fill(~active, 0)
    return terms.sum(), {
        "policy_sum": policy.detach().masked_fill(~active, 0).sum(),
        "kl_sum": kl.detach().masked_fill(~active, 0).sum(),
        "clip_sum": ((ratio - 1).abs() > epsilon).masked_fill(~active, False).float().sum(),
        "tokens": active.sum().detach(),
    }
