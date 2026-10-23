# Gated retraining plan

## Question

Can a target-conditioned drafter trained for greedy accepted-prefix length reduce T5Gemma2 decoding latency while preserving every greedy output token?

## Locked target and data

- Target: base `google/t5gemma-2-1b-1b`, revision `dd0a2683227859151b1730ca3a63087df5b5f39b`.
- Dataset: PubMed train for teacher traces; PubMed validation for diagnosis and benchmark only; test split untouched.
- Generation: greedy, 4,096 input tokens, up to 512 output tokens, `min_new_tokens=32`, repetition penalty 1.05, no-repeat trigram.
- Drafter baseline: current source-conditioned parallel drafter, width 256, 2 layers, block size 4.
- Proposed training: 1,000 CE warm-up steps, then 4,000 updates with a sigmoid top-1 margin accepted-length surrogate plus `0.05 * CE`, sampled anchor windows from target-greedy traces. This is an adaptation for deterministic argmax proposals, not the paper's probability-mass EAL objective.

## Gates

1. **Stop if target path is wrong.** On eight validation examples, exact native-vs-cached target IDs; exact top-1 agreement for full-prefix vs serial cache and block cache at every position tested for K={2,4,8}. Any mismatch blocks training.
2. **Train only after P0.** Collect 1,000 training traces and save a unique drafter checkpoint with the target manifest.
3. **Stop if speculative output differs.** Benchmark 128 held-out validation examples, three repeats; require exact ID sequence for every sample.
4. **Claim speedup only if measured.** Median speed ratio >1.0x, with p50/p95 completion times and exact parity. Acceptance/teacher-forced accuracy by itself is not the result.

## Planned comparisons after the first pass

If the margin surrogate improves top-1 agreement but not latency, compare K=2/4/8 and a DSpark-style sequential Markov logit-bias head plus confidence-based block length. If draft acceptance is still low, train on draft-proposal prefixes or evaluate the public T5Gemma2 DSpark checkpoint as an external positive control. Keep target-only generation as the baseline.
