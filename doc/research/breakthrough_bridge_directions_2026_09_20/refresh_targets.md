# Refresh targets

- Probe entity, number, negation and local n-gram retention at every PPLX tap.
- Verify per-layer memory and side-memory KV caching in generation before a full
  run.
- Measure exact source-token and block recall for reference summaries.
- Record gate values, native/new context RMS, branch-off logit deltas and
  grounded-copy mass for every candidate.
- Recheck validation/test split policy before architecture selection.
- Watch for new 2026 work on dynamic encoder-decoder connectors and
  input-conditioned cross-attention operators.
- Revisit ranking if three-seed direct-projection variance is comparable to the
  expected candidate gain.
