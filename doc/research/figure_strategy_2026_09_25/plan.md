# Figure and research-question decision plan (2026-09-25)

## Decision

Choose one evidence-bearing results figure for the SEAM long-document summarization paper. Assess whether to combine the current RQ1 (reference quality) and RQ2 (source support), and whether the figure justifies a new RQ. This is a figure *plan*, not a claim that the missing experiment has run.

## Existing-work check

- `Paper/src/03_method.tex` already includes an architecture figure.
- `Paper/src/04_experiments.tex` contains four RQs; `Paper/src/05_results.tex` has aggregate scores for PubMed and partial arXiv, but no length- or copy-stratified results.
- `src/evaluation/evaluate_rouge.py`, `evaluate_alignscore.py`, and `evaluate_scale.py` can emit per-example details. Matched prediction files are not presently identified in the workspace.
- Earlier research under `doc/research/bridge_contribution_2026_09_18/` documents risks in lexical source alignment, copy analysis, and long-source evaluation.

## Falsifiable hypotheses

1. **H1 (editorial):** One RQ can cover both reference agreement and source support if its table and narrative keep the metric families distinct. Failure: merged prose treats ROUGE as factuality or hides discordant metrics.
2. **H2 (source access):** A controlled source-budget sweep on matched examples may reveal how SEAM changes as more of a long document is visible. Failure: all systems plateau, or source-coverage differences and summary length explain the apparent pattern.
3. **H3 (copy mechanism):** Character-offset copying may help reproduce source-visible terms when encoder and decoder token boundaries differ. Failure: the main pair has negligible tokenization mismatch, or a no-copy ablation does not change the targeted errors.
4. **H4 (redundancy):** A scatter of aggregate ROUGE and source-support scores adds little beyond the existing table. Failure: it exposes a stable, statistically supported quality/support frontier across completed datasets and systems.

## Source strategy

Read primary ACL papers on context utilization, long-document faithfulness, evaluation metric artifacts, and copying; use official model cards only for tokenizer claims. Audit local model/configuration and evaluator code. Record each external source separately with a short verified quote and explicit relevance/limit. For any external thesis, seek three independent sources; otherwise mark it tentative. Search counterevidence about positional-bias attribution and copying versus factuality.

## Risks and stop criteria

- Existing aggregate scores cannot produce a source-budget curve or copy diagnostic. Do not draw example-level plots from them.
- Model-specific token limits and tokenizer differences can expose different text; log the actual visible source for every system.
- ROUGE and source-support metrics may change with summary length or evaluator context; report length and consider human checks.
- A lexical source match is not proof of factual support; a cross-attention heatmap is not evidence attribution.
- Stop when one recommended figure has an exact RQ, data contract, axes, controls, inference rule, and failure interpretation, supported by primary sources and a local feasibility audit.

## Planned output

`2026-09-25_decision.md`, `sources.csv`, individual source notes, one or more atomic findings, and `refresh_targets.md`.
