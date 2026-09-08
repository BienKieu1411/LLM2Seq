"""Stage-local LR schedules; absent fields preserve the original linear recipe."""

import math


def lr_multiplier(step, total_steps, kind="linear", warmup_ratio=0.0):
    total_steps = max(1, total_steps)
    warmup = min(total_steps - 1, math.ceil(total_steps * warmup_ratio))
    if warmup > 0 and step < warmup:
        return step / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
    return 0.5 * (1 + math.cos(math.pi * progress)) if kind == "cosine" else 1 - progress
