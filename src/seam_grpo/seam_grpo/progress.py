"""Wall-clock optimizer-step timing, including rollout/reward and DDP waits."""

from __future__ import annotations

import time

from transformers import TrainerCallback


def duration(seconds):
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingProgress(TrainerCallback):
    def __init__(self):
        self.started = None
        self.step_started = None
        self.rollout_seconds = 0.0
        self.step_seconds = None
        self.mean_step_seconds = None
        self.completed = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.perf_counter()

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started = time.perf_counter()
        self.rollout_seconds = 0.0

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_started is None:
            return
        self.step_seconds = time.perf_counter() - self.step_started
        self.completed += 1
        # First update may initialize the vLLM host; do not keep that one-off
        # startup cost in later ETA estimates.
        if self.completed <= 2:
            self.mean_step_seconds = self.step_seconds
        else:
            self.mean_step_seconds = 0.2 * self.step_seconds + 0.8 * self.mean_step_seconds

    def metrics(self, state):
        if self.step_seconds is None:
            return {}
        return {
            "progress": f"{state.global_step}/{state.max_steps}",
            "elapsed": duration(time.perf_counter() - self.started),
            "eta": duration(max(0, state.max_steps - state.global_step) * self.mean_step_seconds),
            "update_seconds": round(self.step_seconds, 2),
            "rollout_reward_seconds": round(self.rollout_seconds, 2),
            "train_seconds": round(max(0, self.step_seconds - self.rollout_seconds), 2),
        }
