# g03 — Conformer supports local composition as complementary to global attention

- **Primary source:** Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition* (arXiv:2005.08100, 2020).
- **URL:** https://arxiv.org/abs/2005.08100
- **Web fetch:** 2026-09-23; arXiv HTML lines 3–4, 21–23, 29–49.
- **Role:** Independent architecture precedent for a convolutional local path beside global attention.
- **Credibility:** 5/5 primary paper. **Recency:** older but established. **Transfer:** speech encoders and ASR, not summarization.

**Verified short quote (7 words):** “convolutions efficiently capture the relative-offset-based local correlations”

The paper's stated division is that self-attention learns global interaction
while convolution captures relative-offset local correlation. Its Conformer
block sandwiches self-attention and a convolution module between feed-forward
modules. This supports a narrow architectural claim: an ordered local operator
can supply an inductive bias with a different dependency pattern than global
attention.

The transfer is weaker than g02 because Conformer changes the encoder itself,
uses speech sequences, and has a larger block. It does not show that smoothing
PPLX text states helps abstractive summarization. It is useful as independent
support for the mechanism class and as a warning that locality should be kept
small and residual rather than replacing global retrieval.
