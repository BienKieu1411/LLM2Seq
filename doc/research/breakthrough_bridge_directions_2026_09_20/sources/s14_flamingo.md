# Alayrac et al. 2022 — Flamingo

Source: https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf  
Venue: NeurIPS 2022.

## Verbatim evidence

- Section 2.2: “insert gated cross-attention dense blocks” between pretrained
  language-model layers.
- The added residual is scaled by a learned `tanh` gate initialized at zero, so
  the initial output matches the language model.

## Relevance and limitation

This is the strongest precedent for an independently normalized side-attention
path that cannot corrupt the native token path at initialization. Flamingo is a
multimodal model trained at a very different scale; it supports the interface
and stability rule, not a PubMed ROUGE prediction.
