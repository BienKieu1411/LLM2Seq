# Mechanism-specific SEAM figure — 2026-09-25

## Decision

Do not redraw the overall-score tables as source-length curves or encoder–decoder heatmaps. The main new figure should ask a different, falsifiable question: **Under what source conditions does the bridge help select source-specific content, and does character-offset copy help reproduce that content when the two tokenizers segment it differently?** This can deepen the existing bridge/copy ablation RQ rather than requiring a new RQ solely for a figure.

## Proposed two-panel plot

**Panel A — Where in the source?** x-axis: relative position quartile of a unique, source-visible term or quantity that also appears in the reference summary. y-axis: difference in exact-term recall, `Full SEAM − SEAM without bridge`, in percentage points, with document-bootstrap 95% intervals and a horizontal zero line. Copy remains enabled in both systems.

**Panel B — When tokenization differs?** x-axis: bins of encoder-versus-decoder token-boundary disagreement on the same eligible source spans (define the boundary-set score before viewing results). y-axis: difference in exact-term recall, `Full SEAM − SEAM without copy`, with the same uncertainty display. Bridge remains enabled in both systems.

Use PubMed first, because the full and no-bridge aggregate runs are reported there. The no-copy run and matched per-example predictions are currently missing in this checkout. A small real, verified source→encoder-token→decoder-token→summary example can be placed as an inset if the finished plot needs an intuitive explanation; it is illustration, not quantitative evidence.

## Target definition and controls

Start from names, numeric expressions, and technical terms that appear in the reference and exactly once in the *encoder-visible* source. This avoids ambiguous source positions and selects terms for which exact reproduction is meaningful. Apply normalized exact matching to the generated summary. Audit a sample manually for extraction errors and terms that should be paraphrased, normalized, or omitted. Report the number of terms and documents in each bin, output length, and the rate of unsupported generated names/numbers. Bootstrap by document ID, not individual term, because terms in one document are dependent. Keep training data, budget, checkpoint rule, and decoding matched within each ablation comparison.

This measures **lexical transfer**, not factuality or evidence attribution. If term extraction or tokenizer mismatch leaves bins too small, merge bins by a predeclared rule or do not publish the stratified plot. A negative or mixed difference is a valid result. In particular, the current aggregate PubMed no-bridge row has lower ROUGE but higher AlignScore/SCALE than full SEAM, so the figure must not be captioned as a guaranteed source-support gain.

## Why it adds evidence

The existing RQ3 table says how overall scores change when components are removed. This figure would ask *where* the bridge matters and *when* cross-tokenizer copying matters for the source-specific terms the method is designed to handle. It is not a re-encoding of the same table values. If no-copy and source-span outputs are unavailable, retain only the table; do not fill the figure with synthetic numbers.
