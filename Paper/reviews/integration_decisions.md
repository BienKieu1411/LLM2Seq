# Review integration record

The active manuscript is `Paper/afmr_question.tex` and the proposed system is the `afmr_value_anchor` graph in `src/eviseq_new`. The integration keeps all quality values pending.

## Decisions applied

- The method is described as one full retrieval tensor plus one aligned final-state value tensor. The copy equation now includes the offset-pooled source-prior bias.
- The controller equation now exposes separate source and prompt RMS normalization and the non-negative output-budget clamp.
- RQ1 is operationalized as a source-support signal from AlignScore `nli_sp`; `H=1-C` is a direction transform. Direct entity, number, or relation error claims require an adjudicated audit.
- RQ2 names ROUGE-1/2/L F1 and BERTScore-F1 as the headline statistics. Fairness is defined over records, splits, budgets, preprocessing, update budgets, and decoding; native model input serialization is reported explicitly.
- RQ3 is interpreted as component contribution under the registered decoder and training protocol, rather than universal necessity.
- The four-dataset macro statistic is the unweighted mean of the four dataset-level effects and is secondary to dataset rows.
- The abstract and conclusion say that the evaluation specification is described; they do not imply that quality results already exist.
- The LaTeX preamble protects system, architecture, model, dataset, and metric names with `\\newcommand`; textual macros use `\\xspace` so compiled prose retains word boundaries.
- The agent-reviewed question title now uses ``source support`` as the RQ1 construct, and Abstract A was integrated with evidence-bounded wording. The state lock, story specification, review headers, and main manuscript are synchronized; all scores remain pending.

## Evidence still required

The state audit intentionally remains blocked until the four datasets have fixed manifests and held-out predictions, all registered baselines and ablations have scorer outputs, metric-level paired uncertainty is available, and final model/tokenizer/scorer provenance is archived. BookSum and GovReport preparation records are not present in the current source tree. These are evidence gates, not assumptions to be filled with historical or substitute scores.

The paper compiles with Tectonic and representative pages were visually checked. The local model-free test suite passes; this verifies execution contracts only and does not answer the empirical research questions.

## Plain-English and academic-writing pass (2026-09-13)

The prose in `Paper/drafts/introduction_related_work.tex`,
`Paper/drafts/method_experiments.tex`, and
`Paper/drafts/results_discussion.tex` was rewritten for direct subjects, shorter
sentences, clearer transitions, and less nominal or passive phrasing. The
rewrite preserves the equations, labels, citations, protected terminology,
architecture dimensions, metric definitions, score placeholders, and the
pre-results evidence boundary. The title and abstract were left unchanged
because their exact-candidate audit had already passed.

`git diff --check`, the exact label/citation/number comparison, the academic
writing regression suite, the full Tectonic build, and the local model-free
tests were rerun after the pass. The prose-pattern audit still reports a few
technical-hyphen and template-repetition diagnostics; these are retained where
they name load-bearing terms or repeated result templates.
