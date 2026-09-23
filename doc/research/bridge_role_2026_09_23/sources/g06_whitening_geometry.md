# g06 — Isotropy evidence motivates a diagnostic, but not an automatic bridge

- **Primary source:** Su et al., *Whitening Sentence Representations for Better Semantics and Faster Retrieval* (arXiv:2103.15316, 2021).
- **URL:** https://arxiv.org/abs/2103.15316
- **Web fetch:** 2026-09-23; arXiv HTML lines 3–5, 9–18, and 57–59.
- **Role:** Primary evidence that pretrained language representations can have anisotropic geometry and that whitening can help a retrieval metric.
- **Credibility:** 4/5 primary paper. **Recency:** 2021. **Transfer:** sentence-level semantic similarity, not token-level cross-attention or summarization.

**Verified short quote (12 words):** “the whitening operation in traditional machine learning can similarly enhance the isotropy”

The paper studies BERT sentence vectors, describes anisotropy as a bottleneck,
and reports gains from whitening and dimensionality reduction on semantic
similarity benchmarks. This is evidence for measuring the covariance and cosine
geometry of (P(H_0)) before assuming that the PPLX and Qwen spaces are well
conditioned.

It is weak evidence for inserting whitening into this bridge. Full covariance
whitening would estimate an inverse square root per document, add numerical and
compute cost, and globally mix token rows. A token-wise scale-only normalizer is
especially weak here: decoder cross-attention already applies `memory_norm`
before K/V and grounded copy applies RMS normalization to its memory. A useful
geometry arm would therefore need explicit centering or a learned rotation and
must retain the original values/copy path; otherwise it is likely redundant or
damaging to lexical alignment.
