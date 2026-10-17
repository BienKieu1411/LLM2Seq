# `pure_rouge.py`: implementation audit

Source inspected: `/Users/kieugiangbien/Downloads/pure_rouge.py` (full 2,104-line script). This is local code evidence rather than a published source. The key behavior below comes from the executable path, not the aspirational formula in its module docstring.

## Actual reward and preprocessing

- `normalize_text` only converts whitespace runs to one ASCII space and strips the ends (lines 118–124). It does not canonicalize Unicode.
- `compute_rouge` calls `rouge_score.RougeScorer` for ROUGE-1, ROUGE-2, and ROUGE-L with `use_stemmer=True`, then returns the arithmetic mean of their F-measures (lines 296–360). Empty prediction or reference gives all zeros.
- `clean_completion` removes `<think>…</think>`, prefers the first `<summary>…</summary>`, otherwise the first `<answer>…</answer>`, and removes known tags if neither wrapper matched (lines 127–169). Thus the reward can ignore text that the final evaluator will retain.
- The active `compute_reward` result is `rouge["rouge"]` at line 815. The EigenScore, semantic centering, and Gaussian length terms in its docstring and function body are inactive comments. `EIGEN_WEIGHT` is zero (line 80). The length helper exists at lines 665–700 but is not called on the active path. The required `--embedding-model-path` argument (around line 1473) and stored fields do not make the model part of this reward.

## Training/data behavior that affects interpretation

- Train and eval inputs are independently required as Parquet paths with `source` and `reference`; the loader selects the first `max_samples` rows when requested and does not shuffle that subset (lines 1007–1096). There is no content-level train/eval disjointness check. Supplying the same or overlapping data is possible.
- The script evaluates every 10 steps by default and sets best-checkpoint selection to `eval_reward/rouge` (lines 1235–1252 and 1599–1601). Reusing test data as `--eval-file` therefore leaks it into model selection. The file itself has no test split or safeguard.
- Relative data/output/model paths are rooted at `REPO_ROOT = Path(__file__).resolve().parents[1]` (line 69; resolution at roughly 1845–1868). For the supplied file location under `Downloads/`, that resolves to the home directory, not the LLM2Seq checkout.
- Defaults include full fine-tuning unless `--use-lora` is set; 1,000 steps; train/eval batch 4; four generations; gradient accumulation 1; LR `5e-6`; `loss_type="dr_grpo"`; `scale_rewards=False`; `beta=0`; max prompt/completion 2,048/256; temperature 0.9; top-p 0.95; and vLLM enabled by default (argument/config blocks around lines 1170–1260, 1510–1660). These are the script's settings, not verified compatible settings for EviSeq.
- The script filters both `GRPOConfig` arguments and `GRPOTrainer` constructor arguments through `inspect.signature` (around lines 1350–1425). An unsupported argument can disappear without a warning. Since no TRL version is pinned, in particular the requested Dr. GRPO loss may silently fall back to a different installed default.

## Relevance to eviseq_new

`pure_rouge.py` passes `args.model_name` to stock `trl.GRPOTrainer`; it does not load `EviSeqAFMR`. The custom model in `src/eviseq_new/eviseq_afmr/modeling/model.py` is an `nn.Module` composing its own encoder, a causal decoder with cross-attention, and an adaptive bridge. Its forward pass also carries optional grounded-copy state. Direct reuse of a decoder-only TRL model would not score the same action distribution. For this architecture, rollouts and policy log-probabilities must both include the encoder/bridge and the complete grounded-copy mixture when enabled.

The script is useful as a narrow reference for its active scalar reward and its GRPO configuration intent. It is not an implementation recipe for EviSeq rollout, log-probability, encoder-decoder, or copy-mixture training.
