# F3 — Cross-tokenizer copying is a conditional, more mechanism-specific alternative

## Evidence

Entity- and phrase-copying are already studied in summarization (sources 09–10); character spans are a natural common coordinate for different tokenizers (source 12), but copy behavior is not determined solely by segmentation (source 11). The SEAM paper’s no-copy ablation is still uncomputed (source 08).

## Conditional figure

Possible RQ: “Does the grounded-copy route help preserve source-visible names and quantities when encoder and decoder tokenizations disagree?” First quantify boundary disagreement for the actual SEAM backbone pair. If substantial, score matched test examples for source-visible gold names/numbers under full SEAM versus no-copy. Plot per-category exact-form recall with paired uncertainty, stratified by tokenizer-boundary disagreement. Also count unsupported copied entities; exact surface matching is not factuality.

## Gate / falsifier

Do not elevate this to a headline RQ if the tokenizer audit shows negligible mismatch, the no-copy run remains absent, or the effect disappears after conditioning on term length/type and source visibility. It requires more annotation and design judgment than the source-budget sweep.
