# Karpathy bounded review

The post implementation review used the four `karpathy-coder` checks and a
focused manual pass over the v5-specific model, training, checkpoint and
runner files. The review ran with no model/data download and no CUDA/PubMed
execution.

## Gate results

- Complexity checker, relaxed threshold: 37 files, 9 low-severity warnings.
  They are nesting warnings in copied data/generation paths; no new v5
  objective or module was added to address them because splitting those files
  would increase scope without changing behavior.
- Diff surgeon: 13% noise. The package is new in Git, so docstrings, shebangs
  and blank lines are counted as additions; no drive-by edits outside
  `src/eviseq_update_v5` were found.
- Assumption linter: five documentation warnings remain (four generic action
  words and one verification reminder in a multiline plan step). The goal
  verifier reports strong coverage (22/30, 73.3%) after explicit `verify:`
  checks were added.

The focused review initially found five actionable gaps. They were closed
before the final verification run:

1. Dense and streamed vocabulary loss now share
   `_mixture_nll_from_targets`, with scalar and hidden/lm-head gradient parity
   tests.
2. `Wo=tiny` is calibrated on a deterministic prompt-only batch before fresh
   training; resume skips calibration and restores the saved weights.
3. The PubMed runner evaluates `best.pt` selected by validation CE and sets
   `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so missing local models fail
   instead of downloading.
4. Strict resume checks validate world size, effective batch, scheduler clock,
   training protocol and the checkpoint epoch's canonical manifest hash.
5. Python ROUGE is declared as `rouge==1.0.0`; the route controls are either
   implemented (`detach_copy_route_features`, `hard_source_fallback`) or
   rejected explicitly when reserved (`null_slot`) and unsupported values are
   rejected (`copy_entropy_feature`).

The final local gate is the evidence in
[`IMPLEMENTATION_EVIDENCE.md`](IMPLEMENTATION_EVIDENCE.md): 31 offline tests,
including BF16 projection, legacy-v2 and hidden-interpolation regressions, the
tiny train/evaluate/resume smoke, Ruff lint/format and runner dry-run all pass.
CUDA/NCCL, PubMed ROUGE-1.5.5 and T5Gemma comparison remain registered
research gates rather than claims made by this local review.
