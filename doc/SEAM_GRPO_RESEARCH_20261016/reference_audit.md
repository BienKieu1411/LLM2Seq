# ROUGE reward reference audit for SEAM GRPO

Research snapshot: 2026-10-17. This audit checked the full supplied `/Users/kieugiangbien/Downloads/pure_rouge.py`, the legacy Vietnamese Perl evaluator, the custom `eviseq_new` model, the new GRPO plan/config, and official library documentation/source. It reports implementation facts and a training proxy; it makes no quality or ROUGE-improvement claim.

## Findings on `pure_rouge.py`

The active reward is pure ROUGE: `compute_rouge` scores ROUGE-1, ROUGE-2, and ROUGE-L F-measure with Google `rouge_score` and averages the three; `compute_reward` returns that scalar directly. Eigen alignment, semantic centering, and the Gaussian reference-length factor are commented out. `EIGEN_WEIGHT` is zero. The file's header describes those inactive components, so reading only its docstring would give the wrong formula. See [the implementation audit](sources/refs_pure_rouge_script.md).

The script's active preprocessing first collapses whitespace, removes `<think>` content, and unwraps `<summary>` or `<answer>` blocks. Google `rouge_score` then lowercases, discards non-ASCII-alphanumeric characters, and applies an English Porter stemmer because `use_stemmer=True`. This tokenization is destructive to Vietnamese diacritics and can make canonically equivalent composed/decomposed text score differently. It also creates a second hidden mismatch because the reported evaluator retains wrapper text.

The script requires separate train and eval files but never checks for overlap. It selects the best checkpoint using an eval ROUGE reward and defaults to eval every ten steps, so putting test data in the eval path leaks the test set into model selection. Limited-sample mode takes the first rows. Other important defaults include four candidates, temperature 0.9, top-p 0.95, no KL penalty (`beta=0`), intended Dr. GRPO loss, no reward-standard-deviation scaling, and vLLM enabled. There is no pinned TRL dependency or exact version record, and signature-based filtering can silently omit unsupported arguments. Relative paths are rooted at `pure_rouge.py`'s `parents[1]`; for the supplied Downloads location that is the home directory, not this checkout.

Most critically, this script passes a causal model name to stock TRL. It does not instantiate or adapt the custom `EviSeqAFMR` encoder-decoder graph. Directly wrapping only its Qwen decoder would omit encoder/bridge conditioning and the grounded-copy mixture, changing the policy being optimized. See [the TRL compatibility note](sources/refs_trl_custom_model_compatibility.md).

## Implemented reward for this package

`src/seam_grpo/seam_grpo/reward.py` provides `RougeReward(config)` and a callable `(prediction, reference)` returning `reward`, `rouge1`, `rouge2`, and `rougeL`, each as a float in `[0,1]`. It accepts only the three YAML weight keys, defaults missing weights to 1.0, rejects non-finite/negative weights and rejects an all-zero weight set. The overall score is the normalized weighted mean of the component F-measures.

For Vietnamese, the scorer uses whitespace splitting over Unicode-preserving text. It mirrors the legacy wrapper's prediction preamble-line removal, lowercasing, and bracket remapping. ROUGE-1/2 use multiset n-gram overlap and harmonic-mean F1; ROUGE-L uses token-level longest-common-subsequence F1. It does not stem, strip punctuation, NFC-normalize, or remove `<think>`, `<summary>`, or `<answer>` tags. This makes bad wrapper/reasoning behavior visible to the reward and avoids inheriting the ASCII-only tokenizer.

The result is a fast **single-example training proxy**, not the final Perl ROUGE-1.5.5 score. It uses a single reference string and standard token-level ROUGE-L, with no Perl stemmer, Perl tokenization, corpus/system aggregation, confidence intervals, or multiple-reference handling. The legacy evaluator writes UTF-8, but its Perl script is external; its deployed Unicode behavior cannot be proven from the Python wrapper. Run a small parity check with the exact server ROUGE installation before interpreting training rewards as final-metric values. See [the tokenization note](sources/refs_rouge_tokenization_vietnamese.md).

The finalized YAML and research report use an equal-weight mean of R1, R2, and R-L by default. The implementation does **not** compute ROUGE-Lsum. This target is a configurable experiment choice; it must not be described as an exact match to the final Perl metric.

## Final metric protocol and remaining caveats

The named legacy evaluator `src/evaluation/evaluate_rouge_vietnamese.py` lowercases and remaps historical bracket/quote tokens; it removes lines containing the English preamble from predictions; it supports blank-separated multiple references; and it invokes ROUGE-1.5.5 with `-a -m -n 2 -d`. The training proxy mirrors the visible text cleanup but not Perl scoring internals. The newer `src/evaluation/evaluate_rouge.py` uses a distinct NFC/sentence-splitting protocol and different flags; these wrappers must not be treated as interchangeable.

Group-relative optimization also centers rewards by group. In the current TRL implementation, `scale_rewards=False` means no standard-deviation division, not no centering; tied reward groups therefore have zero policy advantage. The EviSeq trainer must log tied groups and validate its own rollout/current-policy log-probability path, especially for top-p and grounded copying.

## Sources

- [Supplied-script audit](sources/refs_pure_rouge_script.md)
- [Vietnamese ROUGE tokenization and wrapper comparison](sources/refs_rouge_tokenization_vietnamese.md)
- [TRL and EviSeq compatibility](sources/refs_trl_custom_model_compatibility.md)
- [Google Research `rouge_score` tokenizer](https://github.com/google-research/google-research/blob/master/rouge/tokenize.py) and [scorer](https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py)
- [Official TRL GRPO docs](https://huggingface.co/docs/trl/en/grpo_trainer) and [trainer source](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)
- [Independent Python `rouge` implementation README](https://github.com/pltrdy/rouge/blob/master/README.md)
