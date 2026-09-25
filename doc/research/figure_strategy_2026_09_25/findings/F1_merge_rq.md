# F1 — Merge reference quality and source support into one comparison RQ

## Evidence

The current RQ1 and RQ2 already use the same model/dataset comparison and the same main table (`Paper/src/04_experiments.tex`; `Paper/src/05_results.tex`). ROUGE/BERTScore and AlignScore/SCALE answer distinct parts of one empirical comparison. The PubMed result is mixed: SEAM leads on ROUGE but T5Gemma2 is slightly higher on both source-support metrics. The arXiv comparison currently has the reverse source-support ordering; decoder-only results are still missing there. Independent evaluation studies warn that metric agreement and source context are nontrivial (sources 04–07, 14).

## Decision

Rename to one neutral RQ: “How does SEAM compare with fine-tuned decoder-only and encoder–decoder baselines in reference-based quality and estimated source support?” Keep the two metric families visually separated in the table and discuss disagreements explicitly. The merge is an editorial change, not a stronger experimental claim.

## Falsifier / uncertainty

If completed datasets show divergent model rankings across metric families, that strengthens the need for two clearly named *subanalyses* even if there is one RQ. Neither metric family proves factual correctness or coverage.
