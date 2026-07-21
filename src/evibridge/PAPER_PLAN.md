# EviBridge paper plan

## Candidate title

**Beyond Bidirectional Adapters: Evidence-Planned Conversion of Decoder-Only
LLMs for Low-Resource Summarization**

## Claim that can survive review

Do not claim that an external bidirectional adapter itself is new. The paper
claim should be narrower and testable:

> For low-resource summarization, restoring bidirectionality alone is
> insufficient. A summary-specific interface that plans salient evidence while
> preserving full token memory better repurposes decoder-only LLMs as
> encoder-decoder models.

The paper only earns this claim if the full model beats both `causal_ed` and
`lamate_style` under matched backbones, decoder, data and training budget.

## Research questions

1. Does encoder-decoder conversion improve over direct decoder-only
   fine-tuning when both start from the same checkpoint?
2. Does generic token-level bidirectionalization help summarization?
3. Does explicit evidence planning improve ROUGE-2 and factual grounding beyond
   generic bidirectionalization?
4. Is dual memory necessary, or can evidence slots alone retain enough detail?
5. Do the conclusions transfer across Vietnamese and English summarization?

## Minimum result table

All systems must use the same split, prompt, tokenizer, maximum lengths,
training epochs and greedy decoding settings.

| System | Required role |
|---|---|
| T5Gemma fine-tune | External pretrained encoder-decoder baseline |
| Direct Qwen3.5 | Same-checkpoint decoder-only baseline |
| Causal-ED | Conversion without bidirectional planning |
| LaMaTE-style | Generic bidirectional post-encoder baseline |
| Hierarchical | Bidirectional sentence planning without evidence slots |
| EviBridge without oracle loss | Test architectural slots alone |
| EviBridge slots-only | Test hard compression bottleneck |
| EviBridge full | Proposed model |

Report model parameters, trainable parameters, peak memory, training time,
decoding tokens/second, ROUGE-1/2/L, and at least one factuality metric. Use
paired bootstrap confidence intervals for the main ROUGE comparison.

## Go/no-go sequence

1. Run the unit/integration tests. They prove plumbing, not model quality.
2. Overfit 64-128 examples until training summaries are nearly reproduced. If
   this fails, debug before any full run.
3. Run the 0.8B four-system pilot: Direct Qwen, Causal-ED, LaMaTE-style, and
   EviBridge.
4. Scale to the 2B B200 configuration only if EviBridge is better than both
   Causal-ED and LaMaTE-style, preferably by at least 0.5 ROUGE-L or 0.5
   ROUGE-2 with consistent qualitative gains.
5. Compare against T5Gemma only after the 2B EviBridge run is stable.
6. Add an English dataset before submission. A Vietnamese-only result is much
   easier to dismiss as a backbone/tokenizer advantage over English-focused
   T5Gemma.

## What would invalidate the paper direction

- `lamate_style` matches the full model: evidence planning adds complexity but
  no value.
- `causal_ed` matches the full model: conversion and pretrained decoder are the
  real result, not the proposed bridge.
- gains exist only on WikiLingua Vietnamese and disappear in English.
- ROUGE improves only through longer or more extractive outputs while
  factuality decreases.
- T5Gemma comparison uses different data filtering or decoding.

## Expected strengths and risks

The chance of beating T5Gemma is most plausible on Vietnamese datasets because
Qwen3.5 is multilingual and instruction-tuned and the decoder is fully
pretrained. WikiLingua and LR-Sum must be reported separately: the former tests
multi-step procedural selection, while the latter tests single-sentence event
selection and fusion. The harder and more valuable test is English, where
T5Gemma's continued PrefixLM or UL2 pretraining is a major advantage.

The main technical risk is not generation collapse—the pretrained decoder and
gated cross-attention reduce that risk—but that the evidence oracle may be too
extractive. The `no_evidence_loss` and factuality evaluations are therefore
mandatory.

## Cross-dataset protocol

Do not tune evidence granularity independently on every benchmark. Register
three summary-format profiles before the main runs:

1. single-sentence news: sentence units and a fixed budget of up to three;
2. multi-sentence or procedural: sentence units and a reference-length budget
   during oracle-label construction;
3. long documents: non-overlapping groups of three source sentences and a
   reference-length budget.

The reference is used only to create training labels; inference predicts
salience from the source and prompt alone. Keep the bridge, decoder, loss
weights, optimizer schedule, and number of evidence slots fixed across
datasets. A dataset-specific concise output instruction is allowed, but every
baseline on that dataset must receive exactly the same instruction. Report
WikiLingua, LR-Sum, and at least one English benchmark as separate results,
never as a mixed aggregate.
