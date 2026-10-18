# Post-SFT GRPO + gold CE for SEAM

This folder is an independent training phase for the complete architecture in
`../eviseq_new`. It starts from an existing SFT `.pt` checkpoint and its
`resolved_config.yaml`. The SFT training implementation is unchanged.

It uses **TRL 1.16.0's `GRPOTrainer`**, through `SEAMGRPOTrainer`. TRL supplies
the sampler, accumulation/optimizer loop, clipped objective, KL, and policy
metrics. The adapter supplies native SEAM rollout, full-mixture likelihoods,
source conditioning, gold CE, and standard `.pt` checkpoint exports. It overrides
private hooks, so upgrading TRL requires a fresh compatibility audit.

## What is optimized

For every source, sample a group of `G` summaries and compute a weighted mean of
ROUGE-1/2/L F1 against the **training gold summary**. Subtract the group's mean
reward to obtain an advantage for each candidate. Optimize a clipped token-level
policy objective with fixed generation-budget normalization (Dr. GRPO style):

```text
A_i = reward_i - mean_group(reward)
ratio_i,t = exp(current_logp_i,t - rollout_logp_i,t)
L_RL = -sum_valid min(ratio*A, clip(ratio, 1-epsilon, 1+epsilon)*A)
       / (number_of_candidates * max_new_tokens)
L = L_RL + ce_weight * CE(training_gold) + beta * sampled_KL_to_SFT
```

The default is `ce_weight=0.05`, `beta=0`, group size 4, and no group standard
deviation division. These are conservative starting hypotheses, not parameters
proven optimal for this model. Gold CE is evaluated once per original source,
at temperature 1, with prompt and padding labels excluded. It anchors the
reference-summary distribution; it does not guarantee naturalness or factuality.
The CE coefficient is sensitive to the relative gradient scales.

Every sampled token is scored under the **full generator + grounded-copy
mixture**, including the encoder and bridge. Current-policy scoring recomputes
source states with autograd. Rollout caches are cleared before scoring. Inner SEAM
modules stay in eval mode to disable dropout; gradient tracking is enabled for
the loss forwards. The action scorer checkpoints decoder activations and computes
vocabulary logits in small token chunks to reduce memory.

Candidate/gold row selection slices the bridge/copy dataclass tensor fields
inside this adapter, preserving autograd. It also supports original SFT
`BridgeState`/`CopyState` classes that do not implement `index_select` methods.

EOS is included in the action loss. Padding after EOS and decoder prompts are
excluded. Budget-truncated outputs retain their real sampled actions and receive
their measured reward; no artificial EOS or extra length penalty is inserted.
Truncation rate is logged. One optimizer update is made for each freshly sampled
accumulation window; generated trajectories are not reused after the update.

## ROUGE reward versus final evaluation

Training uses a fast, dependency-free, **per-example proxy**: Unicode whitespace
tokens, n-gram multiset F1, and token LCS F1. It mirrors the visible lowercase,
legacy bracket-remapping, and preamble-line cleanup of the Vietnamese evaluator.
It preserves Vietnamese diacritics. It does not use the ASCII-only default
tokenizer in `rouge_score`, and does not silently remove `<think>`/wrapper tags.

This is **not an exact implementation of Perl ROUGE-1.5.5**: tokenization,
stemming, multi-reference matching, sentence-level LCS behavior, and corpus
aggregation can differ. Final claims must use the existing Perl evaluator.
Reward gains therefore need confirmation under that protocol. The supplied
`pure_rouge.py` is used as a reference, not copied as a stock decoder-only TRL
trainer. No EigenScore, judge model, or Gaussian length reward is enabled.

## Configure

Edit `configs/grpo.yaml`:

```yaml
base_config: /workspace/.../sft/resolved_config.yaml
sft_checkpoint: /workspace/.../sft/best.pt
output_dir: /workspace/.../runs/seam_grpo/rouge_ce
```

All paths are resolved relative to the GRPO config file unless absolute. Encoder
and decoder model paths, prompt templates, source token budget, tokenizer fields,
and copy settings come from the SFT config. Model paths must be existing local
directories. Network model loading is disabled. Optional `train_file` and
`validation_file` replace dataset paths without changing prompts. Check that the
splits do not overlap by content/ID: path guards cannot detect duplicated data
stored in different files.
Relative backbone paths are resolved against the SFT config directory. Legacy
CWD-relative paths are accepted if unambiguous; if both locations exist with
different model directories, use absolute paths. Saved configs use absolute paths.

`max_new_tokens: null` inherits the SFT generation budget. Validation uses the
SFT greedy decoding/repetition constraints with that budget. GRPO rollouts use
temperature-only sampling: temperature defaults to 1, top-k is disabled, top-p is
1, no repetition constraint, and no minimum-length EOS mask. The scorer uses
the same temperature as rollout. Truncated-support sampling is intentionally
not exposed because it requires matching support/renormalization in the policy
likelihoods. Validation and serving may continue using their normal settings.

Batch controls are independent:

- `prompt_batch_size`: unique sources per GPU per data microbatch.
- `group_size`: candidates for each source, retained together on the same GPU.
- `gradient_accumulation_steps`: source microbatches per optimizer update.
- `rollout_micro_batch_size`: candidates decoded together with native rollout.
- `vllm_max_num_seqs`: concurrent rollout requests per GPU with SEAM vLLM.
- `vllm_max_num_batched_tokens`: scheduler token budget, automatically raised
  to at least the complete encoder input cap, decoder context cap and sequence
  count. vLLM 0.25.1 disables chunked prefill/MM input for encoder-decoder models;
  a source cap of 8192 therefore requires a scheduler budget of at least 8192,
  even if the decoder only generates 512 tokens. The adapter logs any increase.
- `vllm_encoder_max_batch_tokens`: encoder microbatch token budget per GPU.
- `score_micro_batch_size`: trajectories scored per gradient forward.
- `logprob_chunk_size`: decoder positions in each vocabulary projection.

For 2 GPUs, sources/update = `prompt_batch_size * accumulation * 2`;
candidates/update = sources/update times `group_size`. Reduce rollout/score
microbatch sizes first when memory is insufficient. CE is not multiplied by G.
The schedule is cosine decay with configurable warmup ratio (0 is supported).

## Run

Use Python >=3.11 and the base `eviseq_new` dependencies, then install the pinned
TRL adapter dependency. The example config selects `rollout_backend: seam_vllm`
and requires the server's **vLLM 0.25.1**. Set `rollout_backend: native` to run
without vLLM. From the repository root:

```bash
python3 -m pip install -r src/seam_grpo/requirements.txt

# Validate paths/settings without loading weights.
python3 src/seam_grpo/run_grpo.py --config src/seam_grpo/configs/grpo.yaml --dry-run

# One GPU.
CUDA_VISIBLE_DEVICES=0 bash src/seam_grpo/scripts/run_grpo.sh \
  --config src/seam_grpo/configs/grpo.yaml

# Two GPUs, DDP. Each rank samples and compares complete candidate groups.
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash src/seam_grpo/scripts/run_grpo.sh \
  --config src/seam_grpo/configs/grpo.yaml
```

By default, each run must use a fresh output directory. To intentionally restart
in the same directory, append `--overwrite-output-dir` to the launch command.
This deletes old GRPO checkpoints, metrics and dataset cache; only rank 0 resets
the directory before the other ranks proceed. Keep configs, SFT checkpoints,
backbones and input data outside the output directory. `--dry-run` never resets
it, even when the overwrite flag is present. This is a restart, not a resume.
The flag currently applies to `seam_grpo`.

Initial loading transfers only SFT
weights: optimizer, cosine schedule, and update counter start fresh. Automatic
mid-epoch resume is not implemented. `beta>0` loads an additional complete frozen
SFT model and increases memory use. `beta=0` means the logged KL term is disabled,
not that actual divergence from SFT is zero.

`seam_vllm` creates one colocated TP=1/PP=1 engine on each rank's training GPU,
using vLLM's external launcher and in-process engine. Each rank generates its
complete local groups; DDP synchronizes training gradients. Single-GPU runs use
the same adapter without DDP. This is a native Python engine, not an HTTP host.
No checkpoint file is written for each synchronization: the full actor's CUDA
tensors are copied directly into the colocated engine through `collective_rpc`.
The API/control layout follows pinned vLLM 0.25.1 and TRL's colocate approach.

The model path is currently **Qwen3 causal decoder, cross_attention_every=1,
unquantized TP=1/PP=1**. Encoder families remain those supported by the existing
SEAM vLLM encoder. Other decoder families can use `native` GRPO; the vLLM adapter
rejects them rather than silently switching model architecture.

`vllm_gpu_memory_utilization: 0.20` reserves an engine budget on each GPU. It is
not a cap for training plus rollout combined. Optimizer states, scoring
activations, the optional KL reference and the engine must fit together. Engine
sleep mode and online HTTP transfer are not enabled by this integration.
Full-actor synchronization requires all runtime parameters on one device; CPU
offload is rejected. A missing CUDA device is detected before loading backbones.

## Rollout and weight updates

The policy is the complete encoder, bridge, decoder/cross-attention, and copy
mixture. Native rollout shares the actor object; `seam_vllm` maintains a separate
inference copy with the following synchronized boundary:

1. Hold weights at version k and sample the full accumulation window without
   gradients. Save each action's actual mixture log-probability and EOS mask.
2. Recompute source states and action likelihoods with gradients using all current
   components. TRL computes reward advantages and clipped GRPO; gold CE is added
   once per original source, with a global token denominator.
3. After all accumulation microbatches, update the trainable parameters together.
   Clear consumed rollout buffers and cross-attention caches.
4. For vLLM, verify the full actor manifest, copy all encoder/bridge/decoder/cross/
   copy weights and commit version k+1. Sample the next window, rebuilding encoder states, bridge
   memory/source bias, cross-attention K/V, self-attention K/V, and copy state.

`num_iterations=1` and `steps_per_generation=gradient_accumulation_steps` enforce
fresh trajectories at each update. A version assertion rejects stale input.
`rollout_versions.jsonl` records the boundary; `implementation.json` records the
actual trainer and version. A frozen SFT reference, if enabled, never follows
actor updates. DDP synchronizes gradients for all trainable actor components.

The runtime API is `SEAMLLM.sync_weights(native_model, policy_version=k)`. It
verifies complete parameter/slice coverage, QKV and gate/up packing, tied aliases,
vocabulary padding and shapes. It drains completed request state and invalidates
encoder/copy/KV/prefix-derived state. Inference is locked during sync; partial or
nonfinite updates block further generation until engine rebuild. Ordinary HF
`load_weights` remains initialization-only and rejects unsafe online reloads.

Rollout requests use `processed_logprobs`, so saved probabilities include the
configured temperature. They return sampled token IDs including EOS, without
text retokenization. Dataset sources are not detokenized a second time in the
engine. Each training microbatch measures native-versus-rollout log-probability
MAE/max over valid actions and stops all ranks before backward if configured
tolerances are exceeded (defaults 0.05 mean and 0.50 max, in log units). These
are mismatch guards, not a proof of identical GPU kernels or preserved quality.
Do not raise them to bypass an unexplained mismatch.

The stock TRL `use_vllm` path stays disabled because it would instantiate a
causal-only model. `rollout_backend: seam_vllm` explicitly selects the custom
full-architecture callback while retaining TRL's loss and optimizer loop.

## Checkpoints and evaluation

Outputs include `resolved_config.yaml` for the existing evaluator,
`grpo_config.yaml`, `last.pt`, optional `epoch_001.pt`, and `best.pt` selected by
validation **proxy reward**, never test. By default SFT is evaluated before the
first update; its scores are saved as a baseline. If no epoch beats that score,
`best.pt` retains the SFT weights (epoch/step 0). Inspect `validation_metrics.jsonl`
to see which phase won. Disable `evaluate_before_training` only deliberately.

```bash
CUDA_VISIBLE_DEVICES=0 bash src/seam_grpo/scripts/evaluate.sh \
  --run-dir /workspace/.../runs/seam_grpo/rouge_ce \
  --checkpoint best.pt --split test --batch-size 64 --device cuda:0
```

For the Vietnamese platform's `.txt` source/reference folders, continue using
the existing evaluator with the **new run's** config and checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/rouge155/evaluate_rouge_vietnamese.py \
  --backend afmr \
  --afmr-config /workspace/.../runs/seam_grpo/rouge_ce/resolved_config.yaml \
  --afmr-checkpoint /workspace/.../runs/seam_grpo/rouge_ce/best.pt \
  --model-name seam_vi_grpo --batch-size 64 --device cuda:0 --allow-missing
```

Use a new model name so previous SFT predictions are not reused. Keep final
decoding, source/output budgets, dataset, and ROUGE installation identical to
the SFT comparison. Do not tune reward weights on the seven test sets.

Metrics include individual ROUGE proxies, average reward, gold CE, gradient norm
before clipping, rollout/current-log-prob MAE, clip fraction, optional KL,
completion length, truncation, tied-reward groups, and elapsed time. A policy
loss near zero before the first on-policy update is normal with centered
advantages: gradients can still be nonzero. If every group is tied, the RL term
has no preference signal and only CE/optional KL can contribute.

## Verification and evidence

The policy is the complete conditional SEAM model, not its standalone causal
decoder. For a fixed source and prompts, only generated decoder tokens are
actions; source-token likelihoods do not belong in the GRPO ratio. Gradients
of those action likelihoods still propagate through the encoder, bridge,
cross-attention, value anchor, source bias and copy mixture. Policy scoring
recomputes these states with autograd; detached rollout states are not reused.

When KL is enabled, the frozen reference also owns a complete encoder/bridge/
decoder/copy model and computes its own states for the same source/prompts and
budget. It must not combine a reference decoder with the actor's memory.
`checks/test_conditional_alignment.py` deliberately constructs that incorrect
shortcut and checks that changing the reference encoder exposes the error,
with copy enabled and disabled. It also retains H0, M, value memory, source
bias and copy-state tensors to verify nonzero finite RL gradients directly.

Run `python -m pytest -q src/seam_grpo/checks` in an environment containing the
base model's dependencies. Checks use randomly initialized tiny local backbones,
not downloaded pretrained weights. GPU transfer mapping is checked on torch
fixtures with vLLM's packed layout; vLLM callback/DDP wiring uses an explicitly
marked CPU transport double. CUDA kernels, NCCL execution, VRAM, throughput and
actual ROUGE improvement still require server validation.

Research, source notes, and the adversarial audit are saved in
`../../doc/SEAM_GRPO_RESEARCH_20261016/`. Core references:
[DeepSeekMath](https://arxiv.org/abs/2402.03300),
[Dr. GRPO](https://arxiv.org/abs/2503.20783),
[mixed RL/CE for summarization](https://arxiv.org/abs/1705.04304), and
[pinned TRL GRPO implementation](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_trainer.py).
