# S13 — Cross-encoder attribution versus n-gram matching

**Citation.** Rahimi et al. (2025), *Not Lost After All: How Cross-Encoder Attribution Challenges Position Bias Assumptions in LLM Summarization*, Findings of EMNLP 2025. [Primary paper](https://aclanthology.org/2025.findings-emnlp.846.pdf) · [ACL record](https://aclanthology.org/2025.findings-emnlp.846/).

**Role.** Recent independent evidence that n-gram source attribution can miss semantic correspondences in abstractive summaries.

**Short source quotation (11 words).** “n-gram matching techniques, which fail to capture semantic relationships” (abstract).

## What was tested

The study uses a cross-encoder to attribute summary sentences to source sentences for five LLMs and six summarization datasets, and finds different position-bias patterns from n-gram matching. This concerns evaluation of generated summaries and content position, not training labels for AFMR or PubMed bridge learning. It does not give a suitable coefficient for AFMR's evidence loss.

## Implication for AFMR

Exact phrase overlap is a useful lexical probe but should not be called a verified evidence target. Before training with it, audit false negatives caused by paraphrase and false positives caused by repeated phrases. A semantic aligner might improve label quality, but importing one adds computation and must be separately validated; it is not part of the current implementation.
