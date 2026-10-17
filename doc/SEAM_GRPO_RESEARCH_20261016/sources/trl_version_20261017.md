# Source note: TRL release and GRPOTrainer

- **Checked:** 2026-10-17.
- **Scope:** Latest stable TRL release/tag and official GRPOTrainer documentation/source.
- **Source type:** Primary Hugging Face TRL release notes, docs, and version-pinned source.
- **Credibility:** High for release status and the implementation/documented API.
- **Recency:** Current at the check date; release index and unversioned docs can change.
- **Bias:** Maintainer-authored sources; authoritative for TRL behavior, not independent validation.

## Latest release

The official [TRL releases index](https://github.com/huggingface/trl/releases), checked on 2026-10-17, lists **v1.16.0** first and marks it **Latest**. Its release entry reports **15 Oct, 18:00** and commit `1b5c75e` (the GitHub page omits the year in the date display; 2026 is the year of this check). This is the latest stable release shown by the official repository at the time checked. The [v1.16.0 release entry](https://github.com/huggingface/trl/releases/tag/v1.16.0) links the full changelog from v1.15.0.

## GRPO-related release changes

The [v1.16.0 release notes](https://github.com/huggingface/trl/releases/tag/v1.16.0) state that the fused LM-head path added in v1.15 now runs on devices without Triton (`#7669`). For GRPO, the same release fixes VLM policies generating the image-placeholder token (`#7695`) and fixes `torch.compile` handling of `selective_log_softmax_and_entropy` on Torch 2.6 (`#7719`). It also includes changes to **AsyncGRPO**—tree attention (`#7328`), AMD GPU support (`#7689`), and tool metrics—which are specific to the experimental asynchronous trainer and should not be described as changes to standard `GRPOTrainer`.

The preceding [v1.15.0 release notes](https://github.com/huggingface/trl/releases/tag/v1.15.0) introduced the fused LM head for GRPO and other trainers; they also record a completion-termination fix so every EOS ID declared by a model ends a GRPO completion. The version-pinned [v1.16.0 GRPOTrainer source](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_trainer.py) retains this behavior and exposes the image/video placeholder token handling. The source constructor types `model` as `str | PreTrainedModel | PeftModel`; a custom plain `nn.Module` is outside that declared model contract.

## Official GRPO documentation

- Version-pinned guide source: [docs/source/grpo_trainer.md at v1.16.0](https://github.com/huggingface/trl/blob/v1.16.0/docs/source/grpo_trainer.md).
- Current rendered guide: [Hugging Face TRL GRPOTrainer docs](https://huggingface.co/docs/trl/en/grpo_trainer). The rendered docs describe VLM training and list the model argument as a model ID, `PreTrainedModel`, or `PeftModel`; they also state that instantiated `PreTrainedModel`/`PeftModel` objects must be causal language models. Treat the rendered page as mutable; use the tagged source above for v1.16.0-specific behavior.
- Version-pinned config source: [grpo_config.py at v1.16.0](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_config.py).

The v1.16.0 guide specifies that `scale_rewards=False` disables standard-deviation scaling; the GRPO advantage still subtracts the group mean. It documents `loss_type="dr_grpo"` as an explicit choice, with normalization by a fixed constant corresponding to `max_completion_length`. The current rendered config docs list DAPO as the default loss type, so code targeting Dr. GRPO should set it explicitly and confirm the selected TRL version.

## Evidence excerpts

> “v1.16.0” — the release index lists this tag first and marks it “Latest.”

> “Run the fused LM head and distillation kernels on devices without Triton” — v1.16.0 release notes, GRPO-relevant runtime change.

> “To use this formulation, set `loss_type="dr_grpo"` in the [`GRPOConfig`].” — v1.16.0 GRPO guide source.

