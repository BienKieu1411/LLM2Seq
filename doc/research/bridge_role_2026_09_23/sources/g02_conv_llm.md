# g02 — Qwen3 depthwise convolution gives a direct local-operator precedent

- **Primary source:** Tian et al., *Convolution for Large Language Models* (arXiv:2607.18413v1, July 2026).
- **URL:** https://arxiv.org/abs/2607.18413
- **Web fetch:** 2026-09-23; arXiv abstract and HTML/PDF text, especially Sections 1, 3.3–3.5, 4, and 6.
- **Role:** Closest current primary precedent for adding a tiny ordered local operator to a Qwen-family Transformer.
- **Credibility:** 4/5 primary technical report/preprint. **Recency:** high. **Transfer:** Qwen3 decoder pretraining, not a bridge or summarization experiment.

**Verified short quote (8 words):** “residual depthwise convolution with kernel size k=3”

The report compares 17 insertion points in Qwen3 blocks. Its abstract and
conclusion identify a residual depthwise Conv1D with (k=3), no added
normalization or activation, as the compact design. The full report says the
best placement in its controlled Qwen3 study was on projected Q/K/V before
attention; its micro-ablation favors the residual shortcut and reports that
extra normalization or activation did not improve the selected configuration.

The relevant transferable property is locality, not the reported score. A
depthwise kernel adds (O(kd)) parameters and mixes adjacent positions while
keeping the sequence length. The paper's representation case study reports
that repeated token IDs become more context-sensitive, which is a plausible
mechanism for distinguishing biomedical abbreviations, modifiers, and nearby
numeric units. The paper also notes that some individual benchmark scores fall
and that broader transfer needs more study.

The proposed bridge should not copy the decoder-side placement literally. It
would apply a mask-aware, center-excluded (k=3) residual to the projected
source memory, then leave decoder code untouched. That preserves the source
token indexing and tests a source-side local inductive bias under the actual
PPLX-to-Qwen bridge budget.
