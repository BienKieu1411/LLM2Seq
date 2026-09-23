# Source 07 — Tenney et al. (2019), BERT Rediscovers the Classical NLP Pipeline

- URL: https://aclanthology.org/P19-1452/
- PDF: https://aclanthology.org/P19-1452.pdf
- Authors/venue: Ian Tenney, Dipanjan Das, Ellie Pavlick; ACL 2019.
- Accessed: 2026-09-23.
- Source type: primary empirical probing paper.
- Transfer confidence: evidence that contextual encoders expose lower-level linguistic information; no direct value-bridge result.

## Verified evidence

The authors localize linguistic information across BERT layers and report an average progression from POS and parsing through named entities, semantic roles, and coreference. Their qualitative analysis says BERT can adjust earlier decisions by “revising lower-level decisions on the basis of disambiguating information from higher-level representations.” (12 quoted words.)

## Implication for XOV

This is disconfirming evidence against a simple `the encoder lacks lexical order` story. A contextual encoder can expose low-level lexical/syntactic features and revise them with context, so XOV may duplicate a route already available to cross-attention. It does not prove complete redundancy: the probe studies BERT representation access, not this encoder's cross-attention value projection or decoder-token vocabulary geometry.

## Constraint

Require a matched width-1/no-neighbor control and inspect the change in actual attention-weighted values. If only the lexical bypass helps, describe that result narrowly; do not attribute it to order-aware convolution.
