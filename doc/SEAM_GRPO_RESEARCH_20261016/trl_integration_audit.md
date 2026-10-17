# TRL integration audit for the AFMR encoder-decoder summarizer

Follow-up: optional colocated vLLM rollout now uses the same custom TRL callback
and full-mixture likelihood hook, with a complete actor-update transaction. See
[the vLLM training audit](2026-10-17_vllm_training_audit.md) for current scope and
verification; native-only statements below describe the preceding integration.

Checked 2026-10-17. The active `bienkieu_env` package metadata reports `trl==1.16.0`; the official releases page lists v1.16.0 as latest stable on the check date ([release index](https://github.com/huggingface/trl/releases), [v1.16.0 entry](https://github.com/huggingface/trl/releases/tag/v1.16.0)). The detailed source records are in [`sources/trl_grpo_trainer_v1_16_20261017.md`](sources/trl_grpo_trainer_v1_16_20261017.md), [`sources/trl_fused_lm_head_v1_16_20261017.md`](sources/trl_fused_lm_head_v1_16_20261017.md), [`sources/trl_grpo_config_v1_16_20261017.md`](sources/trl_grpo_config_v1_16_20261017.md), [`sources/trl_grpo_docs_v1_16_20261017.md`](sources/trl_grpo_docs_v1_16_20261017.md), [`sources/transformers_encoder_decoder_docs_20261017.md`](sources/transformers_encoder_decoder_docs_20261017.md), and [`sources/trl_rollout_sampler_issue_7353_20261017.md`](sources/trl_rollout_sampler_issue_7353_20261017.md).

## Finding

TRL v1.16.0 provides the `Trainer.train()` loop, scheduling, generation buffer, group reward/advantage computation, and GRPO objective through the implemented `SEAMGRPOTrainer` subclass (`src/seam_grpo/seam_grpo/trainer.py`). The stock scoring path remains causal-LM-specific: `rollout_func` alone does not change old/current/reference scoring or loss. The adapter supplies a `SEAMActor` wrapper and replaces the rollout and likelihood hooks so scores flow through `EviSeqAFMR`'s separate source encoder, bridge, decoder, and copy mixture. A `GRPOConfig` or rollout callback alone would still score the wrong policy.

## Exact extension surface in v1.16.0

| Need | v1.16.0 hook | What it does and does not do |
|---|---|---|
| Custom sampling | Constructor `rollout_func(prompts, trainer)` | Requires `prompt_ids`, `completion_ids`, `logprobs`; optionally `logprob_token_ids`. Extras are routed to reward functions. It does not supply encoder fields to optimization. The feature is marked experimental and may change. |
| Retain GRPO generation/rewards | Private `_generate_and_score_completions(inputs)` | Builds completion masks, old/reference log-probs, rewards and advantages, then returns a fixed batch. It drops arbitrary dataset/rollout extras from the returned optimization batch after reward calculation. Wrap `super()` rather than reimplementing its group/reward logic, but explicitly append per-completion AFMR source tensors and labels before the result is buffered. |
| Seq2seq token scores | Private `_get_per_token_logps_and_entropies(...)` | The stock signature takes concatenated `input_ids`, `attention_mask`, and `logits_to_keep`; its body calls the model with `fused_lm_head=True` and expects `outputs.log_probs`/`outputs.entropy`. Override this hook to call the AFMR adapter with source tensors, decoder prompt, and prompt-plus-action decoder inputs, then return selected-token log-probs and entropy with shape `[batch, completion_width]`. |
| Keep upstream policy objective | Public Trainer `compute_loss`, private `_compute_loss` | `compute_loss` delegates to `_compute_loss`. Keep `super()._compute_loss` for the clipped/selected GRPO loss, KL, masks, and upstream normalization. Because it does not pass custom encoder fields to the scorer, a small version-pinned `_compute_loss` wrapper may set temporary per-batch encoder context, call `super()._compute_loss`, then clear the context. Avoid copying the full loss implementation. |
| Gold CE anchor | `SEAMGRPOTrainer.compute_loss` around `super().compute_loss`; `SEAMActor.forward(..., gold_selector=...)` | v1.16.0 has no SFT-CE mixing hook. The adapter selects one candidate-slot row per source, retains gold decoder inputs/masks separately from action inputs, and obtains a CE token-sum from the same actor DDP forward. `compute_loss` scales it over unique target tokens across the full accumulation window and ranks. This avoids multiplying the anchor by group size or changing its coefficient by microbatch split. |
| Dataset preprocessing | No dedicated seq2seq collator hook in `GRPOTrainer`'s constructor | Data examples must expose a `prompt` field. Dataset columns can reach reward functions, but `rollout_func` receives only prompt values, not row dictionaries. Encode all rollout-required context in the prompt object or override the generation method to pass raw rows into a custom rollout helper. Preserve source masks, decoder prompt IDs, output budget, and copy-alignment fields through the buffered optimization batch. |

## Model/config wrapper requirements

Passing the current `EviSeqAFMR` directly is not supported: it is a plain `nn.Module` whose `.config` is a Python dictionary. The implementation wraps it in `SEAMActor(PreTrainedModel)`, with a generic `PretrainedConfig` (`model_type="custom"`, `is_encoder_decoder=True`), decoder vocabulary/hidden/token IDs, generation config, and delegated input/output embedding access. The full model is registered at `actor.policy.seam`. The trainer constructs a separate full SFT actor for the frozen reference when `beta>0`, avoiding TRL's default model-ID reconstruction path.

TRL's fused-head helper wraps the actor's `forward`, but dispatches to the original method when `fused_lm_head=True` is absent. The custom likelihood hook does not request the fused path; it builds the prompt-plus-action decoder batch and calls `SEAMActor.forward`, which scores from the full mixture under autograd. The stock scorer must remain overridden because it supplies causal input arguments and expects a `FusedCausalLMOutput`. `model_init_kwargs` are used only when a model ID string is passed and are not a path for this already-instantiated wrapper.

The Transformers encoder-decoder API keeps encoder `input_ids`/`attention_mask` distinct from `decoder_input_ids`/`decoder_attention_mask` ([official model docs](https://huggingface.co/docs/transformers/en/model_doc/encoder-decoder)). For AFMR, each action token's probability must be read from the logits at the preceding decoder-prefix position, conditioned on the same source encoding, decoder prompt, masks, copy state, and output budget used during rollout.

## Sampler/count pitfall

The v1.16.0 rollout-function prose says its prompt slice is unduplicated and asks the callback to return `num_generations` completions per prompt. The actual training sampler uses `RepeatSampler(mini_repeat_count=num_generations)` before `_generate_and_score_completions`; `_generate` passes those rows directly to `rollout_func` and does not expand them again. A first-party issue reports this prose/runtime mismatch ([issue #7353](https://github.com/huggingface/trl/issues/7353)). For the pinned implementation, assert the count and order of rows entering `_generate_and_score_completions`, and return one completion per passed row. Test evaluation separately because its sampler repeats by `num_generations_eval`.

## Behavior-policy probability requirements

The callback's `logprobs` are retained as sampling log-probs, but stock GRPO does not universally substitute them for `old_per_token_logps`; in aligned on-policy mode, the base loss uses detached current log-probs instead. For the custom adapter, explicitly ensure that saved old log-probs and current recomputation use the same model snapshot, tokenizer IDs, decoder-start convention, temperature, logits processors, EOS inclusion, and valid-token mask. Temperature changes probabilities, and top-p/top-k changes support. Start with full-support temperature-only sampling until an on-policy ratio check passes; if using a truncation/filtering transform, score under that same transformed distribution or implement a justified importance correction. Verify `exp(logp_current-logp_old)≈1` before the first update on active tokens.

## Implemented adapter and verification status

`train.py` builds the trainable and optional reference `SEAMActor` instances from the resolved SFT config and checkpoint. `SEAMGRPOTrainer` pins `trl==1.16.0`, calls the native sampler once per row received from TRL's repeated sampler, carries source/copy tensors through the generation buffer, and overrides `_get_per_token_logps_and_entropies` so old/current/reference scores use the SEAM encoder-conditioned mixture. Its `compute_loss` delegates the GRPO objective to TRL and adds the token-normalized gold CE sum. Candidate scoring and CE selection happen within one `SEAMActor.forward` call, which is wrapped by the trainer accelerator for DDP. Rollout and training use the same actor parameters; there is no vLLM weight synchronization path.

`GRPOConfig` explicitly selects `loss_type="dr_grpo"`, the configured reward scaling, full-support temperature-only sampling, and no vLLM. The model wrapper keeps the entire SEAM graph at `actor.policy.seam`; its own `train()` leaves the wrapper in train mode but forces the SEAM tree to eval mode for rollout/scoring parity. The trainer also guards rollout version against optimizer-step changes and keeps a frozen separate SFT reference when KL is requested.

Offline single-process checks on the tiny local model passed, including action log-probability and entropy parity against dense full-mixture logits, policy-gradient flow through encoder/bridge/copy parameters, and SFT-to-GRPO-to-standard-checkpoint reload. The final suite passed all 21 checks, including actual two-rank CPU/Gloo with accumulation and CE enabled/disabled. Gradient and weight equality were checked across ranks. A host-specific macOS MPS barrier issue was fixed in the lifecycle callback before that final run. No pretrained model download or quality claim is involved in these integration checks.

## Verification before any training run

- Construct the subclass without loading remote weights; verify model config, generation config, and forward dispatch.
- In one tiny batch, assert callback input rows = returned completions = rollout log-probability rows; test contiguous group membership on train and eval paths.
- Compare rollout log-probs with recomputed log-probs before updates, including EOS and padding edge cases; check nonzero policy gradient and finite entropy/loss.
- Confirm source tensors survive generation output, `_prepare_inputs` buffering/splitting, and repeated optimizer steps with the correct candidate-to-source mapping.
- Compare CE numerator/token count to a single-source pass; verify exactly one CE contribution per source over each accumulation window under one and multiple ranks.
- Check `beta=0` path first. If enabling KL, verify the custom reference policy loads the SFT checkpoint and returns same-shape log-probs.
- Use a validation split for checkpoint selection; keep test untouched. Report quality only after paired evaluation against SFT under identical decoding and the requested ROUGE-1/ROUGE-2/ROUGE-L evaluator. No gain is implied by this API audit.

## Evidence limits

The architecture/API distinction is supported by primary TRL and Transformers sources: the tagged implementation, config and guide, fused-head utility, and encoder-decoder contract. These are authoritative implementation documents, not independent empirical evaluations. The sampler warning also relies on a first-party issue, corroborated by the v1.16.0 code. There is no official example for this custom AFMR architecture; local tiny offline checks validate adapter mechanics but do not establish improvement in ROUGE or production-scale GPU behavior.
