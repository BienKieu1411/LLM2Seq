# g05 — Nonlinear pre-projection is a known interface family, not a fresh AFMR distinction

- **Primary source:** Shinde, *Position-Agnostic Pre-Projection for Transformer Attention: Nonlinear Feature Construction and Content Skip Before Q/K/V* (arXiv:2604.10791, 2026).
- **URL:** https://arxiv.org/abs/2604.10791
- **Web fetch:** 2026-09-23; arXiv HTML lines 3–4, 9–22, 25–42, and 74–93.
- **Role:** Recent primary precedent for inserting a token-wise nonlinear MLP before Q/K/V.
- **Credibility:** 3/5 primary preprint, single-author and not yet a task-specific bridge study. **Recency:** high. **Transfer:** frozen Pythia probes, not PPLX/Qwen summarization.

**Verified short quote (8 words):** “A pre-projection: a small nonlinear MLP before Q/K/V”

The preprint argues that purely linear Q/K/V projections create a feature
bottleneck and tests an RMSNorm, SiLU residual MLP before Q/K/V. Its probe
experiments report gains on Pythia language-model benchmarks, and the proposed
residual is near-identity at initialization. These results are mechanistic
support for nonlinear feature construction before attention.

The novelty risk is decisive here. AFMR already computes
`feature_up(SiLU(feature_down(RMSNorm(refined))))` and adds it to memory. In the
value-anchor recipe, that branch is already a key-side adaptation while
`value_memory` keeps the final encoder projection for values and copy. A new
controller-free MLP with the same input/output locality would be a simplification
or reparameterization, not a genuinely different bridge role. Prefer the local
position-mixing candidate unless a dedicated ablation is explicitly intended to
test whether the existing branch's controller/gate—not the MLP family—caused its
failure.
