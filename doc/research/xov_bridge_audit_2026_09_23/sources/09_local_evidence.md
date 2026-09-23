# Local XOV code and executable counterexamples

Type: primary local implementation and synthetic experiment. Accessed: 2026-09-23.
Credibility 5/5 for code behavior tested; task efficacy not assessed; advocacy risk medium (author-written probe).

Historical inputs: efce6f8:src/xov_bridge/eviseq_xov/modeling/xov.py;
efce6f8:src/xov_bridge/eviseq_xov/data/source_alignment.py;
efce6f8:src/xov_bridge/eviseq_xov/training/checkpoint.py.
Verbatim code: `residual = self.lexical_up(F.silu(pooled).to(self.lexical_up.weight.dtype)).float()`

This places the nonlinearity after reverse pooling. Source alignment compacts kept IDs and does not carry original token ordinals.
Current eviseq_new/modeling/model.py supplies value_memory to grounded copy when available; historical XOV has separate copy_memory.

New executable: ../probe_repairs.py (relative to report root), results: ../probe_repairs_result.json.
Scope: synthetic categorical readout and isolated bridge, no production decoder run, real tokenizer frequency estimate, GPU/DDP or ROUGE evidence.
