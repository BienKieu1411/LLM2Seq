# ROUGE tokenization and Vietnamese compatibility

Research snapshot: 2026-10-17. The Google Research `rouge_score` tokenizer and the local final Perl wrapper were inspected because the supplied training script imports `rouge_score`, while the project reports Perl ROUGE-1.5.5.

## Google `rouge_score`

The upstream tokenizer lowercases, replaces characters outside ASCII `a-z0-9` with spaces, and drops tokens that fail the same ASCII-only check. Its source contains the pattern `[^a-z0-9]+` in [`rouge/tokenize.py`](https://github.com/google-research/google-research/blob/master/rouge/tokenize.py); the scorer calls this tokenizer and can apply the English Porter stemmer in [`rouge_scorer.py`](https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py). These are primary source files from the library repository.

This creates a serious Vietnamese mismatch. A precomposed diacritic-bearing character is treated as a separator, while a decomposed spelling can leave its ASCII base letter behind. Consequently the score can split or alter Vietnamese words, and canonically equivalent Unicode strings need not receive the same token sequence. Porter stemming is English-oriented and does not fix the issue.

## Local reported metric

`src/evaluation/evaluate_rouge_vietnamese.py` is the explicit legacy protocol inspected here:

- Predictions are read as UTF-8-with-BOM support, stripped line-by-line, and any line containing the exact English preamble `Here is a summary of the text in Vietnamese` is discarded (lines 113–122).
- The Python cleanup lowercases and remaps `-lrb-`/`-rrb-`, brace/bracket variants, and the Penn Treebank quote markers (lines 74–84 and 108–110). It writes UTF-8 text for the external Perl scorer (lines 527–541).
- Blank-separated reference blocks become multiple references, up to seven (lines 125–138 and 531–541). The run uses `-a -m -n 2 -d` (lines 548–550): all systems, stemming, up to bigrams, and detail output.
- The evaluator stores UTF-8 bytes but the actual ROUGE-1.5.5 Perl installation is external to this checkout. Its precise Unicode decoding/tokenization behavior could not be established from the local wrapper, so matching Vietnamese diacritics in the actual deployed Perl runtime still needs a small parity check.

A newer, separate wrapper at `src/evaluation/evaluate_rouge.py` behaves differently: it NFC-normalizes and splits text at sentence boundaries before Perl scoring, and uses `-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m` (lines 18–44, 95–105, and 21–22). It is not the old Vietnamese preprocessing protocol. The implementation here follows the explicitly selected legacy Vietnamese wrapper.

## Implemented training proxy and limits

`src/seam_grpo/seam_grpo/reward.py` applies the legacy prediction preamble-line removal, lowercase conversion, and bracket remapping, then uses Python whitespace splitting, which retains Vietnamese Unicode token contents. It computes multiset unigram/bigram F1 and token LCS F1 for one prediction/reference pair. It performs no ASCII filtering and no stemming. The weighted normalized mean is fast and dependency-free, but it is only a per-example ROUGE-1/2/L proxy: it does not implement Perl's tokenizer, stemmer, corpus/system aggregation, confidence intervals, multiple-reference matching, or the external Perl Unicode behavior.

The existing diagnostic evaluator `src/eviseq_new/eviseq_afmr/evaluation/metrics.py` uses `rouge==1.0.0` and labels itself “not ROUGE-1.5.5” (line 35). Upstream `pltrdy/rouge` likewise documents that its implementation is independent of ROUGE-155 and may differ: [README](https://github.com/pltrdy/rouge/blob/master/README.md). Do not substitute either Python diagnostic score for the requested paper metric.

## Source assessment

- Google Research tokenizer/scorer repository: primary implementation source; high credibility for the behavior of that Python package. The `master` links are mutable, so re-check if the package version changes. Its ASCII tokenizer is not suitable as a Vietnamese token boundary policy.
- Project legacy wrapper: primary source for this project's preprocessing and command flags, but it cannot prove how an external Perl binary handles UTF-8 internally.
- `pltrdy/rouge` README: primary source for the independent Python package's declared scope; useful for rejecting a false equivalence claim, not as evidence for Perl behavior.
