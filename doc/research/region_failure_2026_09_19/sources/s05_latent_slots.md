# S8/S9 — latent evidence slot precedents

Perceiver IO: https://openreview.net/pdf?id=fILj7WpI-g (ICLR 2022). It cross-attends from a compact latent array to a large input, processes the latents, then decodes through output queries. This supports the read–mix–write shape of the evidence-slot bridge. It is a general architecture paper, not evidence that 32 slots improve PubMed summarization.

Set Transformer: https://proceedings.mlr.press/v97/lee19d.html (ICML 2019). Its induced self-attention uses trainable inducing points to mediate interactions at linear rather than quadratic cost in input size. This is another independent precedent for compact latent mediation. It targets permutation-invariant set tasks, so AFMR must retain positional information from its pretrained encoder rather than treat the paper as a direct recipe.

The local candidate differs by conditioning slots on AFMR's prompt-aware controller, writing a bounded residual only to cross-attention key memory, and retaining the full `H0` value/copy path. Those differences are design hypotheses, not established novelty or performance claims.
