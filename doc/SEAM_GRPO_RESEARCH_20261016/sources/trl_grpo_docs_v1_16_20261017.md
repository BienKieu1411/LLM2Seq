# Source note: TRL v1.16.0 GRPOTrainer guide

- **Checked:** 2026-10-17.
- **Source type:** Official, tag-pinned TRL documentation source.
- **URL:** [Tagged GRPOTrainer guide](https://github.com/huggingface/trl/blob/v1.16.0/docs/source/grpo_trainer.md); [current rendered guide](https://huggingface.co/docs/trl/en/grpo_trainer).
- **Credibility:** High for documented usage and public configuration.
- **Recency:** The GitHub source is fixed to v1.16.0; the rendered page is mutable and should not be used to infer an older pin.
- **Bias:** Maintainer documentation explains intended use; implementation remains the authority for runtime details.

## Evidence used

The tagged guide describes the GRPO trainer configuration and supported causal/VLM use. It documents loss-type and reward-scaling choices and the `rollout_func` extension. Use the tagged page with the tagged implementation for this integration, because the unversioned rendered page may move ahead of an installed version.

The rollout hook replaces generation, not the policy scoring/loss pipeline. Its additional fields support reward functions; the guide does not specify a seq2seq scoring callback or a supervised-CE composition hook.
