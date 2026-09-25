# Refresh targets before drawing or submitting

- Locate immutable prediction files, references, IDs, checkpoints, and run manifests for SEAM, T5Gemma2, and chosen decoder-only baselines. Current workspace inspection found no matched test predictions.
- Confirm source text fields, preprocessing, actual tokenizer names and versions, model context windows, prompts, and output caps. Re-audit the dataset-specific configuration values against actual run artifacts.
- Choose sentence-aligned shared word budgets after inspecting source-length and token-fit distributions; report excluded documents and final N.
- Fill remaining main-table dataset/baseline rows before broad four-dataset claims. PubMed/arXiv are the first figure candidates because they already have SEAM/T5Gemma2 aggregate results.
- Audit evaluator versions/window settings and run `--details` for paired analysis; bootstrap by document ID. Clarify whether plotted ROUGE-L is corpus-level or mean per-example F1.
- If pursuing the copy diagnostic, complete no-copy ablation, quantify actual tokenizer-boundary mismatch, and design a source-visible entity/number annotation protocol.
- Recheck any literature published or revised after 2026-09-25, especially studies of long-source factuality evaluation.
