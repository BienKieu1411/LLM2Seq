# Decision: TRL GRPO with a complete SEAM actor

Date: 2026-10-17. This supersedes the original native optimizer loop. Code stays
in `src/seam_grpo`; the SFT architecture and existing vLLM serving code are not
changed by this integration.

Follow-up: the subsequently authorized SEAM vLLM integration is documented in
[the vLLM training audit](2026-10-17_vllm_training_audit.md). Its complete-actor
transaction addresses the online-sync blocker described below. Native rollout
remains available. The earlier 21-test result below records the native-only stage.

## What uses TRL

`SEAMGRPOTrainer` subclasses the actual `trl.GRPOTrainer` from pinned TRL 1.16.0.
It retains TRL's repeat sampler, reward grouping/advantages, clipped policy
objective, optional KL, accumulation and optimizer loop. It overrides rollout,
conditional likelihood scoring, preservation of source tensors, and CE mixing.
The adapter is necessary because the pinned stock scorer supplies causal-LM
inputs and invokes a fused LM head, which cannot represent the custom bridge
and grounded-copy mixture. This follows from the
[tagged implementation](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_trainer.py)
and [fused-head utility](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/utils.py),
with a separate encoder/decoder input contract documented by
[Transformers](https://huggingface.co/docs/transformers/en/model_doc/encoder-decoder).

The runtime, rather than conflicting prose in the rollout guide, determines
callback row counts: TRL's sampler has already repeated each source G times.
The adapter samples one completion per passed row and asserts group alignment.
It explicitly retains the log-probability returned at each actual sampling step
as the behavior probability. Source, masks, alignment and gold inputs travel
through the same buffer shuffle/split as the generated actions.

## Architecture-specific weight lifecycle

The trainable actor is the complete parameter set
`theta = {encoder, bridge, decoder including cross-attention, grounded copy}`.
Only sampled target tokens are actions; source tokens and decoder prefix are
conditioning. At temperature 1 the action probability is the final generate/copy
mixture. At other temperatures it is softmax of the mixed logits divided by the
same temperature used for sampling. Scoring plain decoder logits would optimize
a different policy.

One synchronous optimizer cycle is:

1. Keep theta_k fixed while collecting every candidate group for the complete
   accumulation window, without autograd. Save action IDs, actual behavior
   log-probabilities, EOS-inclusive masks and version k.
2. Compute per-source group-relative rewards from training references. Recompute
   source encoder states, bridge memory/source bias and action likelihoods with
   gradients through the full current model. The denominator in each importance
   ratio is the saved behavior probability; no old hidden state is reused as the
   current conditioning graph.
3. Accumulate the TRL policy objective plus a small gold CE anchor and update all
   trainable components together. Default learning rates are 1e-6 for encoder
   and decoder, 3e-6 for bridge and cross-attention. These are pilot settings,
   not established optima. Gold CE is counted once per original source, with a
   global gold-token denominator over ranks and the accumulation window.
4. Discard consumed trajectories and clear decoder cross caches. The next native
   rollout uses the same now-updated actor object and reconstructs encoder
   states, bridge outputs, cross/self K/V and copy state from theta_(k+1).

`num_iterations=1` and `steps_per_generation=accumulation` enforce this boundary;
the loss rejects stale versions. A frozen full SFT reference is loaded separately
only when beta>0 and is never updated with the actor. DDP synchronizes gradients
for the entire trainable tree. No alternating encoder/decoder update is needed
for the GRPO likelihood ratio to be meaningful: numerator and denominator simply
refer to complete conditional policies on the same source/action history.

The objective is established
[GRPO](https://arxiv.org/abs/2402.03300), with fixed-budget normalization/no reward
std division motivated by [Dr. GRPO](https://arxiv.org/abs/2503.20783). Mixed RL/CE
has a [summarization precedent](https://arxiv.org/abs/1705.04304). Applying these
methods to this architecture does not prove a new GRPO algorithm or a ROUGE gain.

## Why native rollout for now

Using the same actor avoids a separate weight transfer. The existing SEAM vLLM
loader restores its static checkpoint at the end of `load_weights`, which would
overwrite incoming online updates. Its serving path therefore is not currently
an online GRPO synchronization interface. A later vLLM integration needs an
explicit complete-parameter mapping, atomic version boundary, tied-weight
handling and invalidation of encoder/bridge/copy state as well as self/cross KV.
Synchronizing decoder weights alone is insufficient. This conclusion combines
the local loader trace with the documented
[vLLM 0.25.1 transfer contract](https://docs.vllm.ai/en/v0.25.1/training/weight_transfer/)
and [NCCL load path](https://docs.vllm.ai/en/v0.25.1/api/vllm/distributed/weight_transfer/nccl_engine/).
See [the detailed lifecycle audit](rollout_weight_lifecycle.md).

## Verification and remaining evidence

All **21 checks passed** in 28.78 seconds in bienkieu_env on macOS CPU, using
random tiny local backbones. Tests cover dense/chunked full-mixture likelihood
and gradient parity, actual library objective invocation, two successive rollout
versions, frozen SFT reference, standard checkpoint reload, EOS/padding and real
two-process Gloo with accumulation and CE enabled/disabled. An independent
read-only audit found no concrete blocker in grouping, fused-head bypass,
likelihood alignment or CE normalization. Ruff, shell syntax, CLI help and
whitespace checks passed. No pretrained model was downloaded or commit made.

CUDA/NCCL throughput, VRAM, GPU numerical differences and actual ROUGE gains
remain unverified. Reward is a Vietnamese-preserving per-example ROUGE proxy,
not exact Perl ROUGE-1.5.5. Checkpoint selection uses validation, and final SFT
versus GRPO comparison must keep budgets/decoding/data and the deployed Perl
evaluator identical. Small gold CE is an anchor, not a guarantee of natural prose.
