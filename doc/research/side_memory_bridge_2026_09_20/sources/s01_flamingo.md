# S01 — Flamingo

**Primary source:** Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.

URL: https://arxiv.org/abs/2204.14198

## Evidence

- Section 2.1 states that the resampler takes variable-size features and produces 64 visual outputs to reduce cross-attention cost. The paper then uses learned latent queries to cross-attend to the features. The short description is: “produces a fixed number of visual tokens.”
- Section 2.2 says the added cross-attention uses vision-derived keys/values and language-derived queries, and that the layers are gated so the language model remains intact at initialization.
- Section 2.2 gives the mechanism: multiply the added layer by `tanh(alpha)` before the residual addition, with `alpha` initialized to zero. The output therefore initially equals the pretrained model.
- Appendix A.1.1 explicitly concatenates learned-latent K/V with feature K/V in the resampler; this is a precedent for retaining a raw feature route alongside a compact learned route.
- The ablation in Section 3.3 reports a 4.2% overall-score drop and training instability when zero-initialized tanh gating is removed.

## Transfer to AFMR

Use a compact evidence bank `E` produced from `H0`, but add a separate decoder side cross-attention branch. Keep the native token memory `(K0,V0)` and grounded-copy route unchanged. Apply `h <- h + tanh(alpha_l) * Attn(Q(h), K(E), V(E))` after the native token cross-attention, with `alpha_l=0` at initialization. This is structurally different from adding a residual to `K0`.

## Caveats

Flamingo is multimodal, largely freezes its language backbone, and trains on much larger multimodal data. Its gain does not forecast PubMed ROUGE. The result supports the stability and interface pattern, not the task-specific score.

