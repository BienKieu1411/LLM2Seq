# Gao et al. 2026 — Gated Tree Cross-Attention

Source: https://aclanthology.org/2026.acl-long.1629.pdf  
Venue: ACL 2026.

## Verbatim evidence

- The paper introduces a “checkpoint-compatible gated cross-attention side
  branch.”
- Its added structural update is a gated residual rather than a replacement of
  the backbone state.

## Relevance and limitation

GTCA is recent evidence for a controlled side path around a decoder-only LLM.
It uses parse-tree memory, LoRA and staged training on different tasks, so only
the side-path isolation and gate are transferable to the proposed bridge.
