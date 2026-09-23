# Source 01 — Gehring et al. (2017), Convolutional Sequence to Sequence Learning

- URL: https://proceedings.mlr.press/v70/gehring17a/gehring17a.pdf
- Stable landing page: https://proceedings.mlr.press/v70/gehring17a.html
- Authors/venue: Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin; ICML 2017, PMLR 70.
- Accessed: 2026-09-23.
- Source type: primary peer-reviewed architecture paper.
- Transfer confidence: direct mechanism precedent; low task transfer to cross-tokenizer PPLX→Qwen summarization.

## Verified evidence

Section 3.3, Eq. (2), computes attention scores from contextual encoder outputs `z_j`, then forms each attended value as `z_j + e_j`, where `e_j` is the source input embedding at the same source position. The authors write: “We found adding e_j to be beneficial and it resembles key-value memory networks.” (13 quoted words.) They describe `z_j` as carrying a large input context and `e_j` as point information about the particular source element.

## Implication for XOV

This is a direct precedent for a value-only lexical residual while retaining contextual keys. It supports testing a lexical route into decoder-side attention values, but it does not validate reverse alignment: the paper already has one source embedding per encoder position, so no many-to-one scatter or cross-tokenizer boundary problem exists. Its result is from convolutional sequence-to-sequence models trained from scratch on translation and Gigaword, not a frozen/pretrained encoder-decoder with grounded copy.

## Constraint

Use this source to justify the mechanism class only. Do not cite it as evidence that decoder-token embeddings pooled into encoder positions will improve this repository's validation CE or ROUGE.
