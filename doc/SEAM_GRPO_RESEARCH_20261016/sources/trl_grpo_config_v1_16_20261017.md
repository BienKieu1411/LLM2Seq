# Source note: TRL v1.16.0 GRPO configuration

- **Checked:** 2026-10-17; installed package is TRL 1.16.0.
- **Source type:** Official, tag-pinned TRL configuration source.
- **URL:** [Tagged raw `grpo_config.py`](https://raw.githubusercontent.com/huggingface/trl/v1.16.0/trl/trainer/grpo_config.py); [GitHub source](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_config.py).
- **Credibility:** High for configuration field semantics and defaults in this release.
- **Recency:** Exact v1.16.0 tag; defaults are version-sensitive.
- **Bias:** Maintainer-authored configuration, not an empirical recommendation for summarization.

## Evidence used

- `model_init_kwargs` are passed to `AutoModelForCausalLM.from_pretrained` when `model` is a string. If an already-instantiated model is passed, the trainer warns that these kwargs are ignored.
- The v1.16.0 config has `loss_type="dapo"` and `scale_rewards="group"` defaults. Set the intended loss and reward scaling explicitly rather than assuming a different release's defaults.
- `num_generations`, `generation_batch_size`, and `steps_per_generation` jointly constrain the sampler and generation buffering; the config checks generation-batch divisibility by group size.
- `generation_kwargs` are forwarded to the Transformers generation config and can override generation parameters. The custom rollout callback must implement its own generation settings consistently with its probability scorer.

## Direct excerpt

> “Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`”
