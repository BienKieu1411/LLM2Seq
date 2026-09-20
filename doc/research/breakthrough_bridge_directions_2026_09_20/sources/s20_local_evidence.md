# Local code and score evidence

Sources:

- `src/eviseq_new/eviseq_afmr/modeling/afmr.py`
- `src/eviseq_new/eviseq_afmr/modeling/decoder.py`
- `doc/research/region_failure_2026_09_19/final_report.md`
- User-reported `evidence_slots` and `adaptive_topdown` scores on 2026-09-20.

## Observations

- Current AFMR composes encoder taps once, then supplies one memory to every
  decoder layer.
- The decoder already has a copied cross-attention in every Qwen layer and a
  full-token grounded-copy route.
- Direct projection scored `49.671/22.135/45.956`; region/query variants and the
  two later slot/top-down variants were lower on all three ROUGE metrics.

## Limitation

The two latest runs are represented only by aggregate scores. Without paired
predictions, resolved configs and repeated seeds, the shared mechanism is a
design hypothesis rather than a proven causal explanation.
