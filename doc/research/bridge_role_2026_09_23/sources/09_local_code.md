# S09 — Current source implementation and historical experiments

Type: first-party code audit + user-reported outcomes. Accessed: 2026-09-23.
Credibility 5/5 for current code routes; run-comparability confidence limited.
Recency 5/5; advocacy risk medium (this project).
Code: src/eviseq_new/eviseq_afmr/modeling/{afmr,decoder,model,grounded_copy}.py
History: doc/research/bridge_contribution_2026_09_18/{plan,final_report}.md;
doc/research/breakthrough_bridge_directions_2026_09_20/four_directions.md.

Verified excerpt, decoder.py: `hidden = self.memory_norm(memory)`.

Keys use normalized bridge memory; values use normalized value_memory when
provided. Copy preparation uses the same anchored value_memory. AFMR already
has nonlinear feature/depth residuals and focus bias; adding an MLP is not a new
class. Historical contextual-value experiments pool windows; the local optional
variant lacks a separately reported score in that record. Verify git history
before calling any local-context mechanism entirely untried.

No local trained checkpoint was inspected. A missing resolved_config error and
the latest aggregate failure do not establish which architecture or settings
produced the run. Conclusions about why it failed remain hypotheses.

USER CORRECTION: XOV is untested, not a failed experiment. Historical code at
efce6f8:src/xov_bridge/eviseq_xov/modeling/xov.py builds decoder-token lexical
features with rank-256 down-projection, directional depthwise kernel-3
convolution, reverse overlap scatter, SiLU and zero-initialized up-projection.
It adds a bounded residual only to value_memory, with original memory and
copy_memory retained. model.py supplies copy_memory to grounded_copy.prepare.
That code was inspected historically; it is not present in the current src tree.
This is not a fresh end-to-end verification of the historical implementation.
