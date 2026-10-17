# Source note: vLLM async policy changes and cache semantics

**Source:** [vLLM 0.25.1 async RL docs](https://docs.vllm.ai/en/v0.25.1/training/async_rl/). Accessed 2026-10-17. Primary official docs.

**Short quote:** “some tokens in context may still reflect the old weights (stale KV cache).”

**Relevant evidence:** The docs define `pause_generation` modes: abort, wait, or keep queued requests and resume them later. With `keep`, one request may generate tokens before and after a weight update. `clear_cache=True` clears decoder KV/prefix cache; without it cached keys/values can reflect the old model. This is explicit support for async pipelining, but it permits mixed-policy completions.

**Application to SEAM:** Even a cleared decoder KV cache is insufficient for a kept SEAM request. `SEAMModelState` owns request copy state and vLLM's encoder outputs carry packed bridge (M,V,b); those are produced before decoding and are not the standard decoder prefix cache. The custom adapter does not expose a global state reset/versioned request-state API. For single-policy GRPO sequences, wait for current rollouts to finish or abort/discard them before changing actor weights. Alternatively preserve immutable whole-model versions and bind each request, encoder output, cross KV, copy state, and log-probability to one version.

**Limits:** vLLM docs describe generic engine caches, not SEAM's Python-side state. The extra invalidation requirement follows from the local adapter's state and data dependencies.
