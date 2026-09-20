# Lin et al. 2023 — COMPOSITION

Source: https://aclanthology.org/2023.findings-emnlp.108.pdf
Venue: Findings of EMNLP 2023, pp. 1599–1614.

## Verbatim evidence (short excerpt)

- Abstract, p. 1599: “generate specific keys and values passing into different decoder layers.”

The abstract also states that the paper composes representations from different encoder depths. In Sec. 3.2, Eqs. 7–8 define separate weighted combinations for each decoder layer’s keys and values. Sec. 3.1 explicitly gives the standard token-level CE objective.

## Mechanism

The model retains encoder representations from multiple depths. A composed interface creates a distinct K and V sequence for each decoder layer; K and V weights are separate, so the two paths remain aligned in position but need not share the same depth mixture.

## Relevance and limitation

This is the strongest precedent for a PPLX-to-Qwen bridge that uses depth taps without changing the source token positions. The paper is about compositional generalization rather than summarization, so a ROUGE gain is not implied. PPLX can expose intermediate taps in the current code, but a per-decoder-layer memory interface requires extending the decoder wrapper to accept a list of K/V memories.
