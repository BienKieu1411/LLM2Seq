# Prose and notation integrity review

REVIEW PROFILE: Academic-writing integrity and reviewer-facing prose pass.
PAPER: *EviSeq: Source-Grounded Summarization with Composed Pretrained Encoders and Causal Decoders*
SCOPE: The active manuscript rooted at `Paper/afmr_question.tex` and its included drafts, with the proposed implementation restricted to `src/eviseq_new`.
STATUS: Review-only memo; no empirical result or submission-readiness claim is authorized.

## Superseding title and abstract pass

The title and abstract were revised on 2026-09-13 at the author's request.
The current title is descriptive, and the current abstract removes formulas,
score values, and parenthetical metric definitions. The checks and repair notes
below remain useful as a historical integrity record; the active candidate is
`Paper/reviews/active_title_abstract.txt`.

## Strengths

- The title is descriptive and the abstract, Results, Discussion, Limitations, and Conclusion consistently keep quality values pending.
- The four research questions form a readable chain from factual-support signal to quality, component contribution, and encoder transfer.
- The draft distinguishes ROUGE overlap, BERTScore semantic similarity, and AlignScore source-consistency; this prevents a metric from being presented as a factuality guarantee.
- Equations and prose generally use the same names for the AFMR controller, full-source path, copied cross-attention, value anchor, and grounded-copy mixture.
- The manuscript does not use unsupported superlatives or historical scores as evidence.

## Required repairs before a results release

1. Use `provides a route for copying source lexical forms and numbers` instead of `preserves ...`; the latter reads as an outcome or guarantee.
2. State the controller's two input normalizations and the non-negative output-budget clamp in the displayed equation so the prose and implementation agree.
3. State that the graph retains one full retrieval tensor and one aligned final-state value tensor. This is more precise than saying that it has a single memory while also describing separate keys and values.
4. Include the pooled source-prior term in the copy-attention equation and define the index as an eligible aligned source-token row.
5. Define headline statistics once: ROUGE-1/2/L F1 and BERTScore-F1, all on their declared scale; AlignScore native consistency is high-is-better and `H=1-C` is a low-is-better transform.
6. Replace unqualified `matched prompts` language with the exact fairness contract. Native input serialization can differ between EviSeq, decoder-only controls, and T5Gemma2; the records, budgets, split IDs, decoding constraints, and preprocessing should be called matched only where they are actually identical.
7. Keep RQ1's direct entity/number/relation language conditional. The automatic headline is a sentence-level consistency signal; a typed factual-error claim requires a pre-specified audit.
8. Define the four-dataset macro statistic as an unweighted mean of dataset-level effects and label it secondary. Report dataset rows first.
9. Keep all `\tbd{}` cells and declarative templates conditional until the manifests, predictions, scorer outputs, and uncertainty files exist.

## Local style flags

A few sentences use dense hyphenated strings such as `source-grounded`, `decoder-only`, and `train-from-pretrained`. These are standard technical compounds and should be retained where they carry a precise meaning; avoid adding more compound labels during later editing. The repeated protected-name macros in the preamble are intentional and should not be treated as prose repetition. No banned phrase or unsupported claim was found in the active manuscript.

## Verification boundary

This review checks wording, notation, and claim scope. It cannot validate the missing four-dataset runs, dataset manifests, model revisions, scorer parity, or visual PDF rendering. The paper remains a working pre-results draft until those artifacts are produced.
