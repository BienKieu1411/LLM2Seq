# Post-SFT GRPO for a custom encoder-decoder summarizer

Research snapshot: 2026-10-17. Scope: whether and how to test GRPO after SFT on a custom encoder-decoder summarizer, with attention to Dr. GRPO, DAPO token aggregation, supervised anchoring, and rollout probabilities. This is an implementation recommendation for an experiment, not a forecast of ROUGE improvement.

## Finding

The decoder policy in an encoder-decoder summarizer is still an autoregressive categorical policy: at each step it samples a decoder token conditional on the encoded source and the already sampled decoder prefix. The GRPO probability-ratio objective therefore transfers mathematically if the implementation scores those exact decoder actions. However, the recent GRPO/Dr. GRPO/DAPO results are mainly decoder-only mathematical-reasoning experiments. The direct encoder-decoder summarization evidence uses older policy-gradient methods rather than GRPO. Evidence for this exact architecture/task combination is insufficient; treat the following as a small controlled pilot.

GRPO's core comparison is grounded by the original [DeepSeekMath paper](sources/papers_01_deepseekmath_grpo.md), the critique and correction in [Dr. GRPO](sources/papers_02_dr_grpo.md), and [DAPO](sources/papers_03_dapo.md). The best-supported task-specific precedent is [encoder-decoder summarization with mixed supervised and RL loss](sources/papers_05_paulus_seq2seq_summarization.md), complemented by [InstructGPT's auxiliary likelihood objective](sources/papers_06_instructgpt_ptx.md) and [PPO with an SFT KL anchor for summarization](sources/papers_07_stiennon_summary_rlhf.md).

## Recommended first objective

For each source `x`, sample `G` candidate summaries `y_i` from the current behavior policy. Score each full summary with the requested mean of ROUGE-1, ROUGE-2, and ROUGE-L F1 (equal weights by default, configurable if the project needs a different blend). The current local reward is a fast per-summary proxy; it does not claim exact equivalence to the legacy VietnamROUGE-1.5.5 evaluation protocol. Form a group-centered advantage without dividing by that prompt's own reward standard deviation:

```text
A_i = R(x, y_i) - mean_j R(x, y_j)
r_i,t = exp(log pi_theta(y_i,t | x, y_i,<t) - log pi_old(y_i,t | x, y_i,<t))
J_pg = mean_x [ mean_i (1 / L_cap) * sum_t mask_i,t *
                min(r_i,t * A_i,
                    clip(r_i,t, 1 - 0.2, 1 + 0.2) * A_i) ]
L = -J_pg + lambda_sft * L_CE(SFT references) + beta * KL(pi_theta || pi_SFT)
```

`L_cap` is one fixed generation-budget constant used for all samples, with padding excluded by `mask`. Do not divide the policy loss by each sampled response's realized length. This is the Dr. GRPO-style normalization choice; the constant only fixes overall scale. A common batch-wide reward scale may be applied to all `A_i` for numerical stability, but do not use a separate standard deviation per prompt as the initial default. If the reward group is tied, `A_i=0`; count these groups and only add DAPO-style over-sampling/filtering if zero-signal groups materially reduce useful updates.

For the first run, use the supervised cross-entropy term as the retention anchor and set `beta=0`, while logging divergence from the SFT checkpoint. CE should be computed on the training reference summaries with teacher forcing. This supports the summary-writing distribution represented by those references; it does not by itself preserve broad language abilities. If using a learned reward model or observing reward hacking/drift, ablate a KL-to-SFT penalty separately rather than silently stacking multiple strong anchors. The direct sequence-to-sequence study and InstructGPT support auxiliary supervised likelihood, while the summarization RLHF study supports an SFT KL anchor; none establishes the best coefficient for GRPO here.

Use ROUGE as a training signal only if ROUGE is the intended target. It is a proxy: a historical encoder-decoder study found pure RL could have the highest ROUGE while being less readable, and another study evaluated coherence, consistency, fluency, and relevance alongside automatic scores. Keep length visible in the reward/evaluation: avoid rewarding verbosity accidentally, mask or explicitly penalize budget-truncated outputs, and don't impose a blanket shortness penalty if coverage matters. Human review or a validated factuality/coverage evaluator is needed before interpreting a metric gain as better summaries. See [Paulus et al.](sources/papers_05_paulus_seq2seq_summarization.md), [Stiennon et al.](sources/papers_07_stiennon_summary_rlhf.md), and [SummEval](sources/papers_08_summeval.md).

### Initial settings to test, not established defaults

- `G=8` candidates per prompt; Dr. GRPO reports eight, while DAPO used 16 at much larger scale. Begin with 8 if memory permits, and use the same group membership when computing the baseline.
- Sampling `temperature=1.0`, `top_p=1.0`, `top_k` disabled; Dr. GRPO's training settings use this full-support distribution. This makes the first policy-ratio check simpler.
- Symmetric clipping `epsilon=0.2`, one optimizer pass per rollout batch, and a conservative initial learning rate around `1e-6` are hypotheses anchored to reported LLM-RL recipes. For this custom model, bracket the rate around that value and prefer the rate that keeps reward and length stable; do not copy DAPO's large-run batch sizes or response budgets.
- Set the CE coefficient by matching loss scale. As a pilot grid, try `lambda_sft` in `{0, 0.05, 0.1}` after normalizing both objectives consistently, and log the CE and policy-gradient norms. These coefficients are experiment candidates, not literature-backed optima.
- Start with the fixed-horizon Dr. GRPO-style policy reduction. DAPO instead divides by the total valid tokens across a group, so longer outputs receive more share of the update. That may better preserve useful token signals in long reasoning traces, but is an untested choice for short summaries and can make verbosity more influential. Compare the two reductions only after the first run is stable.

## Encoder-decoder implementation requirements

1. Treat the source document as conditioning context, not an action. For every sampled decoder token, save its old-policy log-probability under the rollout distribution. Recompute its current log-probability by teacher-forcing the sampled prefix and conditioning on the same encoder inputs.
2. Keep the tokenizer, decoder-start convention, attention masks, temperature, top-p/top-k filters, and any logits processors identical between sampling and probability recomputation. Include EOS if it was sampled as an action; exclude decoder-start and padding. Reward and loss masks must agree on which output tokens belong to each completion.
3. The importance ratio must compare probabilities from the same behavior distribution. Temperature changes probabilities; top-p changes token support. If sampling uses transformed logits but training recomputes raw-softmax probabilities, the ratio is inconsistent. For top-p, renormalize over the retained support and use the same transform in old and current policy calculations. Begin with top-p 1 until this path is verified. This follows from PPO's behavior-policy ratio and the official [Transformers generation documentation](sources/papers_10_transformers_generation.md).
4. Check that pre-update current and saved old log-probabilities agree on a rollout batch (ratios near 1), then inspect finite losses, group reward spread, EOS/truncation rates, response lengths, KL to SFT, and clip fraction. The published one-update settings are a sensible first limit on policy staleness.
5. If many groups have identical/near-identical rewards, they provide little or no group-relative signal. Log the rate first. Dynamic resampling can help but changes which prompts are trained on; cap retries and report the retained-prompt distribution.

## DAPO vs Dr. GRPO for summaries

These are distinct choices, not a single recipe. Dr. GRPO removes both per-response length division and per-prompt reward-standard-deviation division. DAPO uses a token mean over the group's valid tokens, keeps group-standardized advantages, and adds dynamic sampling, decoupled clipping, and overlong handling. DAPO's own experiments support its full package for long-CoT math, while Dr. GRPO's analysis warns that the two GRPO normalizers can alter response and prompt weights. For short variable-length summaries, start with the simpler Dr. GRPO-style reduction and centered rewards, monitor length, and test DAPO token-mean aggregation as a single ablation. The evidence does not settle which variant is best for summarization.

## Falsifiable hypotheses

1. **Metric-targeting:** On a fixed held-out set and identical decoding, GRPO increases the requested mean ROUGE-1/2/L under the legacy VietnamROUGE-1.5.5 evaluation over the SFT checkpoint. This can fail through reward overfitting, unstable sampling, or mismatch between the lightweight training proxy and the legacy evaluator.
2. **SFT anchor:** Adding a modest summary CE term improves fluency/coherence or limits degeneration relative to pure policy optimization, with a possible cost in reward progress. The direction and best weight are unknown for this model.
3. **Normalization:** Fixed-horizon Dr. GRPO yields less harmful length drift than per-response-mean GRPO; DAPO token-mean may learn useful long outputs better but could over-weight verbose candidates. Track reward at matched length to test this rather than relying on average reward alone.
4. **Probability consistency:** Before any update, recomputed current log-probabilities match stored rollout log-probabilities under identical sampling settings, giving ratios near 1. Failure is an implementation defect, not a training outcome.

## Adversarial review and limits

- **Is longer or higher-ROUGE output necessarily better?** No. Length is a confound and ROUGE can be gamed; one seq2seq study found the pure-RL system with the highest ROUGE was the least readable. Report length, truncation, repetition, factuality, coverage, coherence, and fluency with automatic metrics.
- **Does Dr. GRPO prove that group-standard-deviation scaling is always harmful?** No. Its analysis identifies a prompt reweighting effect. DAPO retains group standardization and reports strong math results, but does not isolate that factor on summarization. The best choice for continuous summary rewards remains uncertain.
- **Does CE preserve language generally?** Only if its examples represent the behavior to preserve. Reference-summary CE anchors summary style and conditional generation; broad capability retention would require an appropriate broader auxiliary distribution or a tested KL constraint.
- **Has GRPO been validated on this architecture?** The token-policy math is architecture-neutral, and older direct summarization work validates RL loss mixtures for encoder-decoders. Recent GRPO experiments do not validate this custom architecture/task; pilot outcomes and stability are unknown.

No ROUGE or quality gain is promised. A clean comparison needs the SFT checkpoint as baseline, the same held-out prompts and decoding, multiple random seeds where feasible, and a small human review set. Report mean and uncertainty, not only the best checkpoint.

## Sources and triangulation

- GRPO mechanism and normalization: [DeepSeekMath](sources/papers_01_deepseekmath_grpo.md), [Dr. GRPO](sources/papers_02_dr_grpo.md), and [DAPO](sources/papers_03_dapo.md); the official [DAPO code release](sources/papers_04_dapo_code.md) supplies implementation context.
- Encoder-decoder and supervised mixing: [Paulus et al.](sources/papers_05_paulus_seq2seq_summarization.md), [InstructGPT PPO-ptx](sources/papers_06_instructgpt_ptx.md), and [Stiennon et al.](sources/papers_07_stiennon_summary_rlhf.md). These support testing anchoring but use different optimizers, models, and reward regimes.
- Summary reward evaluation: [Paulus et al.](sources/papers_05_paulus_seq2seq_summarization.md), [Stiennon et al.](sources/papers_07_stiennon_summary_rlhf.md), and [SummEval](sources/papers_08_summeval.md). These independently motivate checking more than a single automatic score.
- Rollout probabilities: [PPO](sources/papers_09_ppo.md), [Dr. GRPO](sources/papers_02_dr_grpo.md), [DeepSeekMath](sources/papers_01_deepseekmath_grpo.md), and the official [Transformers generation docs](sources/papers_10_transformers_generation.md).

## Refresh targets

Before implementation, recheck the currently installed generation library's post-processor log-probability path, available sequence-to-sequence rollout support, and the project's exact ROUGE tokenizer/normalization. Revisit recent primary work on GRPO loss aggregation and summarization metric-human agreement before changing the objective beyond this pilot.
