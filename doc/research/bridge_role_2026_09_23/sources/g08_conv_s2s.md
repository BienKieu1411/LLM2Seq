# g08 — ConvS2S gives a direct key/value lexical-residual precedent

- **Primary source:** Gehring et al., *Convolutional Sequence to Sequence Learning* (ICML 2017).
- **URL:** https://proceedings.mlr.press/v70/gehring17a/gehring17a.pdf
- **Web fetch:** 2026-09-23; PMLR PDF lines 212–257, especially Eq. (2).
- **Role:** Primary architecture precedent for keeping encoder states as attention keys while adding aligned lexical embeddings to the values.
- **Credibility:** 5/5 primary peer-reviewed paper. **Transfer:** translation and Gigaword summarization, not PPLX→Qwen.

**Verified short quote (7 words):** “We found adding e_j to be beneficial”

Their Eq. (2) computes attention weights from encoder output `z_j`, then
forms the attended value from `z_j + e_j`, where `e_j` is the source input
embedding at the same position. The authors explicitly describe this as a
key/value memory: contextual `z_j` serves as keys and the lexical residual
augments values. This is a direct precedent for testing an ordered lexical
residual before decoder retrieval, but it does not validate the repository's
cross-tokenizer XOV path: XOV obtains `e`-like features from decoder-token
embeddings and reverse-scatter alignment, and that transfer remains untested.
