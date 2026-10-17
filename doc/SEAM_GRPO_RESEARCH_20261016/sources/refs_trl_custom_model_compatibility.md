# TRL GRPO compatibility with eviseq_new

Research snapshot: 2026-10-17. The supplied `pure_rouge.py` does not pin TRL. The repository's `src/eviseq_new/pyproject.toml` pins Transformers but does not declare TRL. The compatibility statements below describe the official current documentation/source checked on this date, not an unrecorded local installation.

## Framework source findings

- Current official [TRL GRPOTrainer documentation](https://huggingface.co/docs/trl/en/grpo_trainer) accepts a model string, `PreTrainedModel`, or `PeftModel`; it specifies that model objects currently supported by the trainer are causal language models. The docs state: “Only causal language models are supported.” This scope does not directly admit the project's custom encoder-decoder `nn.Module`.
- The current [TRL GRPOTrainer source](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) centers rewards within each generation group. `scale_rewards=False` disables standard-deviation division but does not remove group-mean centering; equal rewards within a group therefore yield all-zero advantages. The official [trainer docs](https://huggingface.co/docs/trl/en/grpo_trainer) describe `False`/`"none"` as no std scaling and `dr_grpo` as fixed-constant token-loss normalization. These are distinct operations.
- Current TRL docs show a default loss type of DAPO, while Dr. GRPO is an explicit loss type. `pure_rouge.py` requests `loss_type="dr_grpo"` but filters config fields to the installed constructor signature. With no dependency pin or runtime-version record, a filtered-out option could leave an unintended default.

## Architecture finding

`src/eviseq_new/eviseq_afmr/modeling/model.py` defines `EviSeqAFMR(nn.Module)`, with a source encoder, causal decoder with cross-attention, and adaptive residual bridge. `encode_source` prepares bridge memory and optional grounded-copy state; `forward` passes that state through the custom decoder. Sampling is implemented in `eviseq_afmr/evaluation/generate.py`, not through the ordinary causal model's `generate()` path. The GRPO action probability must be the exact distribution used by this generation code, including the grounded-copy mixture when enabled.

The supplied script passes a model identifier to `GRPOTrainer`, so TRL loads a standard Transformers model rather than this graph. It also enables vLLM by default. Neither setting adapts the trainer to the custom encoder, bridge, copy distribution, or custom generation routine. The repository's current implementation addresses that mismatch with `SEAMActor(PreTrainedModel)` plus a pinned `SEAMGRPOTrainer` subclass: it stores native rollout token IDs and behavior log-probabilities, then recomputes current-policy probabilities through the same EviSeq path and sampling law. This replaces the earlier native-loop fallback recommendation while retaining TRL's trainer loop and objective.

The external-generation defaults in `pure_rouge.py` include temperature 0.9 and top-p 0.95. A custom trainer must compare rollout and current probabilities under the same temperature and support truncation; recomputing raw-softmax probabilities for top-p samples would use a different behavior distribution.

## Source assessment

- Hugging Face TRL docs/repository: primary framework documentation and source; high authority for current behavior. The `en` docs and `main` branch are mutable, and no version was recovered from the supplied script/project manifest. Record the exact `trl` version and verify the constructor/loss behavior before any TRL reuse.
- Local EviSeq AFMR source: primary source for the project's actual architecture and generation API. The conclusion against direct stock TRL use follows from its `nn.Module` contract and full custom forward path; it is not an assertion that encoder-decoder GRPO is mathematically invalid.
