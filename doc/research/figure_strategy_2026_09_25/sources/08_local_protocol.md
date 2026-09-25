# Local SEAM protocol audit (2026-09-25)

- Primary paths: `Paper/src/03_method.tex`, `Paper/src/04_experiments.tex`, `Paper/src/05_results.tex`, `src/eviseq_new/configs/afmr_base.yaml`, `src/eviseq_new/configs/afmr_arxiv.yaml`, `src/eviseq_new/configs/afmr_booksum.yaml`, `src/eviseq_new/configs/afmr_govreport.yaml`, `src/T5Gemma/configs/`, `src/decoder_baselines/configs/suite.yaml`, and `src/evaluation/`.
- Type: local first-party code and manuscript; inspected 2026-09-25.
- Credibility 5/5 for what the checked-in files specify; recency 5/5; low-bias 3/5 for performance claims because run artifacts are absent.
- Short code quotes: `max_source_length: 4096`; `max_source_length: 8192`; `max_source_length: 12288`.
- Facts: the paper already includes `figures/architecture_connected.png`; RQ1/RQ2 share one aggregate table. SEAM and T5Gemma2 configs set dataset-specific source caps to 4096/8192/12288 tokens, but different tokenizers may expose different source text. ROUGE, AlignScore, and SCALE evaluators have `--details` for per-example outputs keyed by ID. The visible workspace contains smoke training logs but no test predictions or dataset files for drawing the proposed new plot.
- Limit: checked-in configs do not prove which settings were used by every scored run. Resolve each checkpoint's run manifest before a claimed comparison.
