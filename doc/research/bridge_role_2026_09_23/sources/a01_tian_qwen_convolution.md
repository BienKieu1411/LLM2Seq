# Source A01 — Convolution for Large Language Models

- Authors: Yuchuan Tian et al.
- Version: arXiv:2607.18413v1, 20 July 2026
- URL: https://arxiv.org/html/2607.18413v1
- Type: primary technical report; Qwen3 language-model pretraining study
- Quality: high for the reported Qwen3 ablations, medium for transfer to PPLX→Qwen summarization (different task, training regime, and insertion point)

## Verified evidence

The abstract reports a controlled comparison of 17 insertion points in a Qwen3
Transformer. The authors select a residual depthwise Conv1D with kernel size 3,
without added normalization or activation, after the QKV projections. In their
table, the pre-QKV location (P4) improves the reported Qwen3-1.7B WikiText
perplexity from 13.42 to 13.06; the post-QKV location (P5) is stronger at 12.85.
The report explicitly says the study is on Qwen3 models trained from scratch,
so these numbers do not forecast a bridge added between independently pretrained
PPLX and Qwen checkpoints.

Short verbatim excerpts (under the source quote limit):

> “a residual depthwise convolution with kernel size k=3”

> “without additional normalization or activation”

## Relevant ablations and caveats

- Table 1 (Section 3.2) places P4 after RMSNorm and before QKV; P5 operates on
  the concatenated QKV outputs. P5 is the report’s selected location, but a
  bridge-only change can reach only a pre-decoder-memory analogue of P4 without
  editing `decoder.py`.
- Table 2 says the residual shortcut is better than plain convolution, while
  pre-normalization and sandwich normalization are worse than the shortcut.
  This supports keeping the existing decoder normalization and adding no bridge
  normalization in the first probe.
- Table 4 finds zero convolution weights with a zero or random bias can train,
  while setting both weight and bias to zero is less effective than random
  initialization in their from-scratch setup. This does not decide the
  zero-initialized residual used here; it only warns that initialization must be
  measured rather than assumed.
- Section 4 studies one WSC example and states that nearby context can
  differentiate repeated token IDs. The authors also caution that a single
  cosine-similarity example is not a general mechanism proof.
- Section 5 reports average benchmark gains but several individual benchmark
  scores decrease. The report itself limits its conclusion to the evaluated
  Qwen3 configurations.

## Use in this audit

This is positive mechanism-level evidence for testing a tiny position-preserving
local operator. It is not evidence of ROUGE improvement, factuality, or a
benefit after PPLX contextualization. The proposed bridge should therefore use
only the low-risk pre-QKV analogue: a three-tap depthwise residual on source
states, with the original H0 value/copy path preserved. The post-QKV result
cannot be imported without changing decoder cross-attention.
