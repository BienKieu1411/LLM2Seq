# S08 — Local implementation audit

**Type:** first-party implementation evidence, checked 2026-09-18. **Credibility 4/5, recency 5/5, bias risk 2/5.** This is code inspection and tiny-model testing, not a full PubMed outcome.

**Files:** `src/eviseq_new/eviseq_afmr/modeling/afmr.py`, `modeling/model.py`, `modeling/decoder.py`, `modeling/grounded_copy.py`, `training/engine.py`.

**Short source quotation:** “The direct projection is the controlled ``w/o bridge`` variant.” (`afmr.py` comment.)

The full graph creates AFMR-adapted memory, unmodified `H0` value memory, and a source bias read by both decoder cross-attention and grounded-copy attention. All major bridge output factors start at zero, so the initial function is very close to direct projection. The new auxiliary loss uses the predicted source bias and train-only weak labels; it has a direct gradient path to the focus scorer. Tiny tests verify nonzero gradients and parameter update, but cannot establish ROUGE improvement.
