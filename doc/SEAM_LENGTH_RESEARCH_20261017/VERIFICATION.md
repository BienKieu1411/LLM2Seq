# Verification evidence

Date: 2026-10-17. This verification applies to research artifacts and the narrow numerical script audit, not GPU training.

- Ran `checks/audit_supplied_scripts.py` with `/Users/kieugiangbien/bienkieu_env/bin/python` successfully. Selected pure functions were extracted through AST; model imports and training entrypoints were not run.
- Confirmed the supplied `lead_len_control.py` ignores length differences at controlled quality 0.4, but responds to them at 0.6. The actual constants are 1.0 ROUGE and 0.0 Eigen.
- Checked Unicode/NFKC examples against the evaluator's actual counter. `Việt Nam đang phát triển` is 5 with the evaluator and 8 in the default LARFT mode. Decomposed accents also expose missing normalization in `lafrt.py`'s Unicode mode.
- Saved SHA256 of all six supplied scripts and twelve inspected integration files; no model checkpoint was opened.
- Ruff check and Ruff format check passed for the audit script.
- Source notes distinguish inspected full texts, abstracts, author code and local evidence. Proceedings/journal/preprint status was verified at the linked primary pages. Author code is not counted as an independent replication of its paper.
- Adversarial review covered infeasible token caps, conflicting gold CE, mixed-length groups, mutable dual state, reward sign/dead zones, multi-reference length targets and unsupported encoder-decoder superiority.
- Agent checks were unavailable due to usage limits. This report does not claim independent agent agreement.

Not verified: actual SEAM learning curves, full seven-dataset benchmark, runtime overhead, GPU memory, new target-word conditioning in native/vLLM, or 1-/2-GPU training of the proposed mode. No new mode is advertised as runnable.

## Dual-prompt update

Three `luna_worker` agents completed literature, code-path and adversarial reviews; see `reviews/dual_prompt_review.md`. Primary sources for CTRLsum, GSum, LAAM, entity planning and two-sided length conditioning were opened and scoped in separate notes. Additional author-code documentation is not counted as independent confirmation of the paper. Earlier wording about unavailable agents applies to the initial report only. No new model tests or runtime edits were performed in this research update.

## Custom reward update

The same three agents completed follow-up reviews on reward feasibility and failure modes. New primary sources distinguish ROUGESal, document coverage, CPMI decoding, the attention interpretation debate, and TRAC's abstract-level PRM description. `checks/reward_formula_checks.json` records synthetic arithmetic examples and group-centering cancellation only; it does not evaluate teacher likelihoods or model outputs. The inspected runtime-file fingerprints remain unchanged.
