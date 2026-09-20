# Ivison et al. 2023 — HINT

Source: https://aclanthology.org/2023.acl-long.631.pdf  
Venue: ACL 2023.

## Verbatim evidence

- HINT uses encoded instructions to produce “parameter-efficient modules
  inserted into an underlying model.”
- The generated modules include layer-aware prefixes/adapters rather than full
  model weights.

## Relevance and limitation

HINT supports a hypernetwork that converts one encoded input into bounded,
layer-specific modules. It conditions on tasks/instructions and uses separate
pretraining; a per-document cross-attention operator for summarization remains a
new hypothesis.
