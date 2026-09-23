# g04 — Qwen-team evidence for where nonlinearity breaks attention's low-rank path

- **Primary source:** Qiu et al., *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free* (NeurIPS 2025).
- **URL:** https://papers.nips.cc/paper_files/paper/2025/file/904e89bb4e632e75fb47f093b620b257-Paper-Conference.pdf
- **Web fetch:** 2026-09-23; PDF Sections 3–4, especially lines 64–70, 280–364, and Table 3.
- **Role:** Primary evidence that a nonlinearity between value and output projections changes a low-rank attention path.
- **Credibility:** 5/5 peer-reviewed primary paper from the Qwen team. **Recency:** 2025. **Transfer:** trained 1.7B/15B LMs, decoder-internal gates, not a source bridge.

**Verified short quote (10 words):** “introducing non-linearity upon the low-rank mapping in the softmax attention”

The paper rewrites the value and output projections as a low-rank linear map,
then compares gates at several positions. Its value-position gate (G2) and
SDPA-output gate (G1) improve over the baseline in the reported Qwen-family
language-model experiments, while gating after the final dense projection does
not address the same bottleneck. The paper's analysis also reports that
query-dependent output gating outperforms source/value-only gating, which is a
direct warning for a static source transform.

This supports the *possibility* that a nonlinear source memory can add a
function unavailable to a single linear K/V projection. It does not make a
per-token MLP bridge novel: current AFMR already contains a SiLU low-rank
feature branch. It also does not support a guaranteed ROUGE gain, because its
training objective, placement, and query-dependent gate differ from this task.
