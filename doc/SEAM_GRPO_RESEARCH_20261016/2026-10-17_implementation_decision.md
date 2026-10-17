# Decision: a separate post-SFT GRPO phase for SEAM

Update: the initial native optimizer loop has been replaced by a real,
version-pinned TRL `GRPOTrainer` subclass. The current decision and verification
are in [the TRL rollout decision](2026-10-17_trl_rollout_decision.md). Research
motivation below remains applicable; any statement about the original loop is
historical.

## Implemented decision

New code lives in `src/seam_grpo`. It imports the SFT architecture from
`src/eviseq_new`; the existing model, SFT trainer, and vLLM code are not edited
by this task. It loads the complete SFT state dict and starts a fresh optimizer,
step counter, and cosine schedule. Checkpoints retain the original architecture
contract, so existing evaluation can load them.

The loss combines a clipped, group-centered policy objective with gold-summary
CE. Fixed generation-budget normalization follows Dr. GRPO; reward-std scaling
is disabled by default. Rollouts use full-support temperature sampling, with
the same temperature applied when recomputing action likelihoods. The old log
probability is saved at the actual sampling step, rather than reconstructed
from a different decoding distribution. One fresh rollout window is used for
one optimizer update.

The complete copy/generate mixture defines the action policy. Encoder source,
bridge controller, memory, source bias, decoder cross-attention, and grounded
copy are recomputed with gradients for each scoring microbatch. Source tokens
and decoder prompt tokens are conditioning context, not optimized actions.
EOS participates in the loss; post-EOS padding does not. Rollout and CE use
the same output budget for the bridge controller. Rollout K/V caches are
cleared before gradient scoring, and dropout is disabled consistently.

Native rollout is chosen for correctness of current model weights. The vLLM
host presently loads a fixed checkpoint; using it in online GRPO would require
validated per-update weight synchronization and sampling-log-prob parity.
Simply attaching a stock decoder-only TRL trainer would omit the custom bridge
and copy distribution. This limitation is a throughput cost of the initial
implementation, not a claim that native generation is universally preferable.

## Research basis and limits

The group baseline and clipped objective come from
[DeepSeekMath](sources/papers_01_deepseekmath_grpo.md). The fixed-budget and
no-reward-std choices are motivated by
[Dr. GRPO](sources/papers_02_dr_grpo.md); [DAPO](sources/papers_03_dapo.md)
provides a different token-normalized alternative. Those experiments mainly
concern mathematical reasoning. Their numerical results are not evidence
that this summarizer will improve.

Gold CE as an anchor has a direct summarization precedent in
[Paulus et al.](sources/papers_05_paulus_seq2seq_summarization.md), while
[InstructGPT](sources/papers_06_instructgpt_ptx.md) and
[summarization RLHF](sources/papers_07_stiennon_summary_rlhf.md) give related
retention mechanisms. These support testing an anchor, not a universal
`ce_weight`. The initial 0.05 coefficient, learning rates, and group size 4 are
pilot settings. CE supports the behavior represented by the gold summaries;
it does not prove fluency, factuality, or broad language retention.

The supplied script's active reward is mean R1/R2/R-L, despite inactive Eigen
and length-penalty descriptions in its header. Its default Google ROUGE
tokenizer is ASCII-oriented. The new scorer preserves Vietnamese characters
and mirrors the visible legacy evaluator preprocessing, but is a single-pair
whitespace/LCS proxy. It does not reproduce every Perl ROUGE-1.5.5 behavior.
This is a material metric limitation: validate gains with the unchanged final
Perl evaluator. Details and source evidence are in
[reference_audit.md](reference_audit.md).

## Controls and rejection criteria

1. Measure the SFT checkpoint on validation before updates, using the same
   greedy settings as subsequent validation. Save this baseline; if no GRPO
   epoch exceeds it, best.pt retains the SFT weights.
2. Never use the seven test datasets to choose reward weights, CE coefficient,
   learning rate, or epoch. Different paths alone do not prove content-level
   split separation; ensure the data are disjoint upstream.
3. Watch tied reward groups, truncation, completion lengths, CE, gradient norm,
   and sampled/recomputed log-prob error. With beta=0 the reference KL is
   disabled; the logged zero is not evidence of no drift from SFT.
4. Before a long run, use a small train/validation subset with G=4, then compare
   CE weights 0 and 0.05. Reward gain accompanied by worse reference CE or
   degraded text is a reason to stop or strengthen anchoring. Learning rates
   and coefficient scales require validation, not only loss magnitudes.
5. Compare final SFT and GRPO predictions with identical input/output budgets,
   preprocessing, and decoding. Report the final metric and inspect examples;
   an increase in proxy reward alone does not establish success.

## Adversarial review

- Could the system optimize a different policy? This is tested by comparing
  chunked likelihoods and every parameter gradient against dense original
  mixture logits at multiple temperatures, plus rollout likelihood parity.
- Could zero-looking policy loss imply no learning? Group centering can make
  the scalar cancel before an update while gradients remain nonzero. A test
  checks the reward-gradient sign and a separate RL-only weight update.
- Could CE accidentally train sampled text? CE uses the untouched collator gold
  inputs once per source; candidates have separate action tensors and masks.
- Could a higher score just reflect length or reward-tokenization artifacts?
  Yes. Truncation and lengths are logged, but length-matched/human evaluation
  and exact server Perl evaluation remain necessary. No length or Eigen reward
  is silently inherited from commented code.

## Verification

See the final test summary in `src/seam_grpo/VERIFICATION.md`. Tests use only
randomly initialized tiny local backbones. CUDA/NCCL, full-model memory, runtime,
and ROUGE improvement remain unverified until the server run. No model weights
were downloaded, heavy dependencies installed, or commits created.
