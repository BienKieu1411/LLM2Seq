"""Token-level TIS for the custom, synchronized SEAM rollout callback."""

from __future__ import annotations

import math

import torch


def correct_vllm_policy_inputs(inputs, logps, behavior_logps, mask, *, cap: float, loss_type: str):
    """Match TRL's token_truncate correction without its causal-only vLLM loader.

    There is one optimization iteration per fresh rollout window. The version
    guard prevents an optimizer step inside that window, so current.detach()
    is the old *training* policy, not the separate inference policy. Reuse this
    forward instead of scoring the entire generation window a second time.
    """
    delta = (logps.detach().float() - behavior_logps.detach().float()).masked_fill(~mask.bool(), 0.0)
    # Clamp before exp: a corrupt very large ratio must not overflow first.
    correction = delta.clamp(max=math.log(cap)).exp()
    inputs["old_per_token_logps"] = logps.detach()
    inputs["importance_sampling_ratio"] = correction
    if loss_type != "vespo":
        advantages = inputs["advantages"]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        # The supported TRL policy losses are homogeneous in A. Weight A to
        # apply TIS only to the policy term, leaving KL, entropy and gold CE
        # exactly as upstream. VESPO consumes importance_sampling_ratio itself.
        inputs["advantages"] = advantages * correction
    return correction
