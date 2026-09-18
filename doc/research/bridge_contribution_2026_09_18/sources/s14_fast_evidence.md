# S14 — Query/source fusion for evidence extraction

**Citation.** *Fast Evidence Extraction for Grounded Language Model Outputs*, FEVER 2024. [Primary paper](https://aclanthology.org/2024.fever-1.24.pdf).

**Role.** Related evidence that query-conditioned encoding improves selection of relevant source units, with explicit efficiency trade-offs.

**Short quotation (12 words).** “finds an intermediate point to include query-source cross attention” (PDF p. 2).

## Claim and evidence

The paper compares Early-, Mid-, and LateFusion for extracting evidence spans corresponding to an abstractive output. Its Medical Dataset Table 4 (PDF p. 6) reports EarlyFusion precision/recall 81.29/83.16 and MidFusion 76.37/82.18; the authors discuss that MidFusion trades some extraction quality for throughput, while query-conditioned EarlyFusion performs best in that row. This is a **different task**: evidence extraction after generation, not training AFMR's summarizer. It supports the importance of query/source interaction but not a quantitative ROUGE forecast.

## AFMR implication

Use the current decoder hidden state as the query to a small region memory instead of applying only one source-level prior to every summary token. Keep the full source accessible and measure added training/evaluation time.
