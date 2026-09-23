# g07 — RMSNorm explains why scale-only bridge normalization is a poor candidate

- **Primary source:** Zhang and Sennrich, *Root Mean Square Layer Normalization* (NeurIPS 2019).
- **URL:** https://proceedings.neurips.cc/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf
- **Web fetch:** 2026-09-23; paper abstract and Section 4.
- **Role:** Primary normalization reference used to audit the current decoder/copy path.
- **Credibility:** 5/5 primary peer-reviewed paper. **Recency:** foundational. **Transfer:** normalization properties transfer; no PPLX/Qwen bridge experiment.

**Verified short quote (9 words):** “RMSNorm regularizes the summed inputs to a neuron”

RMSNorm removes mean-centering and retains scale normalization. In the local
implementation, each cross-attention layer normalizes bridge memory before its
key/value projections, and grounded copy normalizes its memory-derived context.
Therefore a bridge change that only rescales each token is largely hidden from
both downstream reads. A sequence-level center or rotation is different, but
that is a higher-risk global operation and needs an explicit geometric
diagnostic before training.
