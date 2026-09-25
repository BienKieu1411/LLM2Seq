# F4 — Adversarial review of the proposed figure

1. **Token fairness.** Equal `max_source_length` does not mean equal source text under different tokenizers. Require the same sentence-aligned prefix and verify it fits every encoder or decoder prompt; otherwise the x-axis is mislabelled.
2. **Document-composition confounding.** Do not compare different document bins as though their mean-score differences were the effect of context length. Vary prefix budget *within* the same document IDs. Stratification by natural length may be reported only descriptively.
3. **Evaluator leakage / context shift.** Score against the unchanged full source with the same evaluator windows at every budget. Publish the evaluator configuration. Long-document factuality scores are not human verdicts (sources 05–07, 14).
4. **Reference/outcome confounding.** A model may get higher ROUGE by outputting longer summaries; log generated length and use the same output limit (source 04).
5. **Causal overreach.** A budget curve tests response to visible source text, not whether the bridge/copy modules cause that response. A bridge/copy ablation or controlled evidence intervention is needed for component-level claims. Lexical matching alone is unsafe evidence-position attribution (source 03).
6. **No post-hoc winner selection.** Decide datasets, budgets, score families, and comparison models before seeing curves; show non-improving outcomes too. No source predicts a SEAM gain.
