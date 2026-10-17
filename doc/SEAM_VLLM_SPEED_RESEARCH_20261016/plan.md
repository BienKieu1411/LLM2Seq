# SEAM vLLM exact-speed audit

Date: 2026-10-16

## Question

Find practical inference-speed improvements for the SEAM vLLM 0.25.1 adapter
without changing model weights, precision, token budgets, or copy semantics.

## Evidence and limits

- Inspect the current adapter and the user's OOM log.
- Compare the custom copy-logit path directly with the AFMR reference.
- Read official PyTorch and version-pinned vLLM documentation/source.
- Ask three read-only reviewers to inspect copy arithmetic, runner state, and
  encoder/bridge tensor lifetimes.
- Run local CPU parity tests, formatting/static checks, and the adapter test
  suite. The Mac has no CUDA runtime, so do not claim server speed or GPU parity.

## Decision rule

Keep only changes with unchanged reference math and a focused test. Treat
batch-size/microbatch changes, kernel/backend changes, and CUDA graphs as
server experiments until predictions are compared at the target GPU. Preserve
all max-input and max-output limits.
