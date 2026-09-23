# Source 06 — Ethayarajh (2019), contextualized representation geometry

- URL: https://arxiv.org/abs/1909.00512
- Authors/venue: Kawin Ethayarajh; EMNLP-IJCNLP 2019.
- Published version: https://aclanthology.org/D19-1006/
- Accessed: 2026-09-23.
- Source type: primary empirical representation-analysis paper.
- Transfer confidence: evidence against a simple static-embedding redundancy argument; no direct bridge experiment.

## Verified evidence

The study compares ELMo, BERT, and GPT-2 representations across layers. It finds that same-word representations become more context-specific in upper layers and reports: “less than 5% of the variance in a word's contextualized representations can be explained by a static embedding for that word.” (21 quoted words.)

## Implication for XOV

The result cuts both ways. A decoder vocabulary embedding is not a replacement for the contextual encoder state, so a lexical value residual can carry point information that H0 does not encode in the same form. At the same time, the result does not show that the residual is useful after the encoder has processed the source; it only rejects the stronger claim that contextual states are equivalent to a static embedding. XOV should therefore be evaluated as a finite-budget access path, not as recovery of information known to be absent.

## Constraint

Keep the direct encoder path and compare XOV against a matched lexical-only width-1 branch. A result from width 1 would support lexical access, while a further gain from width 3 would be needed before claiming ordered local composition.
