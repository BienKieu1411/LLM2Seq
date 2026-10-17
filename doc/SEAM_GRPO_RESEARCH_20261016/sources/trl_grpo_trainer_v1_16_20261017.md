# Source note: TRL v1.16.0 `GRPOTrainer` implementation

- **Checked:** 2026-10-17; installed `bienkieu_env` metadata also reports TRL 1.16.0.
- **Source type:** Official, tag-pinned Hugging Face TRL implementation.
- **URL:** [Tagged raw `grpo_trainer.py`](https://raw.githubusercontent.com/huggingface/trl/v1.16.0/trl/trainer/grpo_trainer.py); [GitHub source](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_trainer.py).
- **Credibility:** High for the exact v1.16.0 implementation and declared API.
- **Recency:** Current pinned release at check date; private hooks can change in later releases.
- **Bias:** Maintainer code is authoritative for behavior, but it does not independently validate custom architectures.

## Evidence used

- The constructor declares `model` as `str | PreTrainedModel | PeftModel`; its docs say instantiated `PreTrainedModel` and `PeftModel` inputs must be causal language models.
- `rollout_func(prompts, trainer)` is an experimental sampling hook. It must return `prompt_ids`, `completion_ids`, and `logprobs`; it can optionally return `logprob_token_ids`. Other rollout fields are forwarded to reward functions.
- `_get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False, compute_aux_loss=False, pixel_values=None, image_grid_thw=None, num_images=None, pixel_attention_mask=None, spatial_shapes=None, num_tiles=None, image_sizes=None, token_type_ids=None, mm_token_type_ids=None, image_position_ids=None)` constructs causal `input_ids`/`attention_mask` model inputs and calls `model(..., labels=labels, fused_lm_head=True)`. It returns per-token log-probs, optional entropy, and optional auxiliary loss.
- `_generate_and_score_completions(self, inputs)` calls `_generate`, builds concatenated prompt/completion IDs, and invokes the log-probability helper for old/reference scores when needed. It merges rollout extras into dataset rows for reward calculation, then returns a fixed tensor dictionary; arbitrary extras are not copied to that optimization batch.
- `compute_loss` delegates to private `_compute_loss`. `_compute_loss` concatenates prompt and completion IDs and calls the helper without encoder-side source inputs. It then applies the configured GRPO loss and normalizers.
- `_prepare_inputs` owns generation buffering, splitting, and reuse. `_get_train_sampler` uses `RepeatSampler(mini_repeat_count=num_generations)` and the evaluation sampler similarly repeats by `num_generations_eval`.

## Direct excerpt

> “A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.”

The source supports retaining TRL's loop and GRPO objective through subclassing, but no documented hook alone changes the causal input/scoring contract to an encoder-decoder policy.
