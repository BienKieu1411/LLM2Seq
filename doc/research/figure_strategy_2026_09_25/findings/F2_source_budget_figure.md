# F2 — A matched source-budget curve is the most useful next results figure

## Evidence

Source length and source position affect long-document summarization in prior original studies (sources 01–02, 13). The SEAM draft claims an interface that preserves access to valid encoder positions but does not yet show whether increasing visible source text helps or harms the final summarizer. The existing main table is one operating point per dataset, and local evaluator scripts can save per-example detail (source 08).

## Figure / RQ

New RQ2: “How does increasing the amount of source text available at inference affect SEAM and the baselines?” Use a full-width 2×2 small-multiple plot: PubMed and arXiv as columns, ROUGE-L and SCALE as rows. x = a nominal word budget implemented with the *same longest sentence-aligned prefix not exceeding the budget* for every model on each held-out document; y = score on the same reference and full original source. Put SEAM, T5Gemma2, and a decoder-only baseline on each panel once all predictions exist. Show document-bootstrap 95% intervals and sample size. Report actual mean visible words per budget. Report AlignScore in a companion appendix panel/table to check that the source-support story does not depend on a single metric.

Choose budget values only after auditing all tokenizers and model windows. Select one fixed set of sufficiently long documents that remains identical at every budget; prevent any model from silently truncating the supplied prefix. Keep checkpoint, prompt format, decoding, evaluator source/chunking, and output length cap fixed. Record actual visible words/tokens and generated length. Recompute every condition, including the nominal full-budget endpoint, with the same inference/evaluation scripts.

## Interpretation / falsifier

If the SEAM–baseline gap increases as more shared text becomes visible, that is compatible with better use of added context, *not proof* that the bridge retrieves evidence at distant positions. A flat or decreasing gap is equally informative and must be reported. Metric shifts could be caused by output-length changes (source 04), evaluator-context effects (sources 05–06, 14), or the source contents introduced at each step. Validate any strong factuality interpretation on a human-checked subset (source 07).

## Feasibility boundary

No matched per-budget predictions or dataset files were found in this checkout. Aggregate Table 1 values cannot be converted into this curve. This is a proposed experiment/figure, not a completed result.
