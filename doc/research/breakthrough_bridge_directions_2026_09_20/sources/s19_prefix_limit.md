# Petrov et al. 2024 — Limits of prefix tuning

Source: https://openreview.net/pdf?id=JewzobRhay  
Venue: ICLR 2024.

## Verbatim evidence

- Prefix tuning “cannot change the relative attention pattern over the content.”
- It can instead bias an attention output toward a low-rank prefix subspace.

## Relevance and limitation

This is negative evidence against choosing a source-conditioned prefix as one of
the primary four directions. A prefix path may steer an existing Qwen behavior,
but the current failure calls for changing how Qwen aligns to PPLX tokens. The
document-conditioned operator is therefore preferred over a prefix-only bridge.
