# T5Gemma2 drafter: research and gated retraining

## Why we are not training first

The existing smoke runs do not establish either exact speculative decoding or a useful draft model. The PubMed Colab smoke run used 8 training articles, 4 validation articles, 32 updates, 4 draft tokens, and only 64 output tokens. Its target-only baseline matched Transformers, but speculative output matched the target on only 2 of 4 examples; the median raw speed ratio was 0.874x. A separate pilot used CNN/DailyMail, 128 training examples, one epoch, and only 4 parity examples; it also reached 2/4 exact sequence matches. The two results are separate and should not be combined.

Greedy speculative decoding must return the same target tokens even with a poor drafter. A failed output parity test therefore points to the verifier/cache/policy path, not to too few training steps. The new `diagnose-verifier` command compares full-prefix, token-cached, and block-cached logits on the same decoder prefixes before any trace collection or optimization.

## Research conclusion

1. Exact speculative decoding keeps the target model's output unchanged by verifying draft proposals using target distributions. On T5Gemma2, the documented decoder API accepts `EncoderDecoderCache`, decoder masks, and cached decoder inputs; those semantics need a direct test at the actual runtime version. See [S01](sources/S01_speculative_decoding.md) and [S02](sources/S02_t5gemma2_cache_api.md).
2. There is already a public DSpark drafter trained specifically against the base `google/t5gemma-2-1b-1b` verifier. Its author reports a 3-layer draft, 200K mixed-domain examples, six epochs, and 146,486 steps. The card reports 40.55% teacher-forced acceptance for its smaller checkpoint, but runtime speed is workload-dependent: its own batch-size-one table includes wins and regressions. This is a useful reference point, not an independent reproduction or a PubMed result. See [S03](sources/S03_t5gemma2_dspark_card.md).
3. The recent EAL paper optimizes expected accepted prefix length using the draft probability assigned to each target-greedy token and distinguishes that objective from the sampling-oriented WTV objective. Our runtime proposes each draft token with deterministic `argmax`; under that policy, the exact acceptance signal is a discontinuous top-1 match. We therefore use a sigmoid target-vs-best-competitor logit-margin surrogate after a small CE warm-up. This is a training adaptation, not the paper's probability-mass EAL objective, and its sigmoid values are not calibrated acceptance probabilities. See [S04](sources/S04_acceptance_aware_training.md).
4. DSpark adds within-block Markov bias and per-position confidence on top of a block-parallel draft. These are plausible next architecture changes if the smaller EAL-trained draft still has low acceptance, but they must be compared by end-to-end latency, not accepted-token rate alone. See [S05](sources/S05_dspark_algorithm.md).
5. DistillSpec supports the broader idea of aligning a drafter on its own on-policy continuations and selecting losses to match the decoding task. This motivates a later proposal-prefix training ablation; it is not part of the first gated retrain. See [S06](sources/S06_distillspec.md).

## Experiment plan

**P0: verifier correctness.** Use the unmodified base T5Gemma2 checkpoint, pinned to the recorded revision. Compare native generation with cached target-only generation and compare full-prefix logits against serial cached logits and block-cached logits. Test eight PubMed validation documents, 32 decoder inputs per document, and block sizes 2, 4, and 8. Require exact top-1 equality at every tested position and exact target-only output IDs. If any comparison fails, stop and fix that path before training.

**P1: drafter training.** Use only PubMed train documents to collect 1,000 target-greedy traces. Train the existing 256-wide, two-layer source-conditioned drafter for 5,000 updates, using 1,000 CE warm-up steps followed by the sigmoid top-1 margin accepted-length surrogate plus a 0.05 CE auxiliary term. Validation and test targets are not read for training. The model/config stay on the GPU host; the local Mac does not fetch target weights.

**P2: held-out verification and speed.** On 128 PubMed validation inputs, run native and speculative paths in alternating order with four warm-ups and three measured repeats. Require exact full sequence parity for every request. A useful result needs median speedup above 1.0x; report p50/p95 per-request latency and accepted tokens as diagnostics. If parity passes but speed does not, reject this drafter as a speedup and then test a larger DSpark-style Markov/confidence draft.

The runnable path is `src/encdec_drafter/scripts/retrain_gated.sh`; the separate config template is `src/encdec_drafter/configs/t5gemma2_pubmed_margin.yaml`. Both require a local base checkpoint and CUDA on the execution host.

## Execution status on 2026-10-23

The verifier diagnostic, margin-surrogate objective, config, and gated runner are implemented and locally tested. The model was not trained in this session: the Mac has no CUDA GPU, no T5Gemma2 checkpoint (the cache is 32 MB of metadata), no `nvidia-smi`, and no Colab CLI or connected Colab runtime. The configured checkpoint path is intentionally a placeholder. No model weights were downloaded locally. The actual GPU run remains pending execution on Colab or the user's server.
