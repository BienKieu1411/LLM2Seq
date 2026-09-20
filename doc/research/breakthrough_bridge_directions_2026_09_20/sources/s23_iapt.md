# Zhu et al. 2024 — IAPT

Source: https://aclanthology.org/2024.acl-long.771.pdf  
Venue: ACL 2024.

## Verbatim evidence

- IAPT installs a soft-prompt generator “at each Transformer layer.”
- The generated prompts depend on the current input instruction.

## Relevance and limitation

IAPT supports layer-specific, instance-conditioned adaptation with small
generators. It changes prompt representations rather than Q/K/V operators and
does not test long-document summarization, so the transfer remains speculative.
