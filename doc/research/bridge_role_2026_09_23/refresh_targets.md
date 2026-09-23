# Refresh targets

- XOV status: UNTESTED, explicitly confirmed by user on 2026-09-23.
- Recover exact historical implementation/config only if implementation is
  requested; do not overwrite the user's current architecture automatically.
- Before running: trace visible source/overlap offsets, memory normalization,
  shared embeddings, masked convolution, DDP empty alignment and cache reorder.
- Collect matched direct/full/XOV resolved configs, validation predictions,
  ROUGE per example, residual norms and second-update gradient checks.
- Revisit lexical-value hypothesis after real XOV results; do not infer them
  from contextual-value, evidence-router or delivery-ledger results.
- Novelty review: ConvS2S Eq2 lexical values, cross-tokenizer interfaces and
  ordered local value composition. Papers support mechanisms, not promised gains.
