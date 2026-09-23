# Source A03 — GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

- Authors: Joshua Ainslie et al.
- Venue: EMNLP 2023, pp. 4895–4901
- URL: https://aclanthology.org/2023.emnlp-main.298/
- Type: primary peer-reviewed attention-capacity study, including summarization
- Quality: high for GQA/MQA trade-offs in T5; indirect for Qwen cross-attention in this repository

## Verified evidence

The paper defines GQA as an intermediate number of key/value heads between
multi-head attention and single-head MQA. It reports that MQA can degrade
quality, while uptrained GQA comes close to MHA quality at near-MQA speed. The
study applies GQA to both decoder self-attention and cross-attention and
evaluates CNN/DailyMail, arXiv, PubMed, MediaSum, and MultiNews among other
tasks. Its Table 1 reports GQA-8 quality close to MHA-XXL and generally above
MQA-XXL on the listed summarization aggregate.

The paper also says that key/value heads are mean-pooled when converting a
multi-head checkpoint and require extra pretraining to adapt. That is evidence
that reducing KV head capacity can be a real quality constraint, even though
GQA is often a good trade-off.

Short verbatim excerpts (under the source quote limit):

> “MQA can lead to quality degradation”

> “GQA achieves quality close to multi-head attention”

## Code mapping and implication

`src/eviseq_new/eviseq_afmr/modeling/decoder.py:61-64` reads
`num_heads`, `num_kv_heads`, and `head_dim`. Lines 82–84 project memory into
`num_kv_heads` K/V heads, and lines 117–146 call SDPA with GQA or explicitly
repeat both K and V across query-head groups. Thus each query head in a group
shares the same source-value head; only its attention weights differ.

This argues against a first experiment that rewrites the value memory. A
K-only local bridge can test address geometry while retaining the existing H0
values and grounded-copy features. If the true bottleneck is value capacity,
K-only cannot fix it; that is a falsifier to measure, not a reason to alter the
decoder or copy head under the current scope.

The GQA paper is an encoder–decoder T5 study, not a PPLX→Qwen bridge study.
Its PubMed scores cannot be transferred to AFMR.
