# Source note: rollout prompt duplication report in TRL issue #7353

- **Checked:** 2026-10-17.
- **Source type:** First-party repository issue with a user-provided reproduction; implementation behavior was independently checked in the v1.16.0 tag.
- **URL:** [TRL issue #7353](https://github.com/huggingface/trl/issues/7353).
- **Credibility:** Medium for the reported symptom; the issue is not itself normative API documentation.
- **Recency:** Opened 2026-09-23, shortly before the v1.16.0 check.
- **Bias:** Single-user report and reproduction; treat as a warning, not proof. The v1.16.0 `RepeatSampler` and `_generate` code establish the actual call path.

## Evidence used

The report identifies a mismatch between the rollout-function prose (“no duplication”) and training inputs already expanded by the GRPO sampler. In v1.16.0, the training sampler's `mini_repeat_count` is `num_generations`, then `_generate` passes the current prompt rows directly into `rollout_func`. Therefore, assert the actual batch sizes and return one completion per row received; do not multiply by `num_generations` without checking the version-specific call path. Evaluation uses a separate sampler with `num_generations_eval`.

## Direct excerpt

> “The reward function 'reward' returned 8 rewards, but 4 were expected”
