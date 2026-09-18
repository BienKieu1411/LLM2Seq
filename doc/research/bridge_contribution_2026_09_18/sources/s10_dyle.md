# S10 — DYLE dynamic evidence during decoding

**Citation.** Mao et al. (2022), *DYLE: Dynamic Latent Extraction for Abstractive Long-Input Summarization*, ACL 2022. [Primary paper](https://aclanthology.org/2022.acl-long.118.pdf).

**Role.** A concrete precedent for evidence weights conditioned on decoder history, with a counterexample to treating the mechanism as a guaranteed ROUGE gain.

**Short source quotation (10 words).** “dynamic snippet-level attention weights during decoding” (abstract, PDF p. 1).

## What was tested

DYLE trains an extractor and generator together. At each output token the generator assigns weights to extracted source snippets, conditioned on prior output tokens, and marginalizes token probabilities over them (pp. 1–3). It also trains extractor targets using an oracle and a consistency loss. Thus its dynamic source selection is much more than an extra cross-attention bias; its gains cannot be assigned solely to dynamic weighting.

In Table 4 (PDF p. 6), the paper reports arXiv ROUGE-1/2/L of 46.41/17.95/41.54 for DYLE and 48.24/20.26/41.78 for LSH. The latter is a different system, not a controlled DYLE ablation. Table 5 ablates auxiliary objectives on GovReport and QMSum, not arXiv. The arXiv result is a **cross-system counterexample** to universal superiority, not proof dynamic weighting causes a decrease.

## Implication for AFMR

A decoder-query-conditioned region prior has a plausible role where a source-only prior cannot distinguish summary stages. Its benefit and cost in AFMR remain untested. Do not import DYLE's published gain, or its oracle/consistency losses, as evidence for the proposed AFMR variant.
