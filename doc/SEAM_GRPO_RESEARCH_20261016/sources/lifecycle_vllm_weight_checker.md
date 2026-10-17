# Source note: vLLM weight checker coverage

**Source:** [vLLM Weight Checker docs](https://docs.vllm.ai/en/latest/training/weight_checker/), accessed 2026-10-17. Primary official docs; latest API may be newer than the locally pinned vLLM 0.25.1.

**Short quote:** “All actions cover the target model's parameters” (8 words).

**Relevant evidence:** The checker hashes each model parameter per inference shard and supports checksum, reset, and compare while the engine is paused. Its documented limitations say buffers, draft models, and LoRA adapters are not checked.

**Application to SEAM:** Parameter hashes can catch an incomplete/misdirected tensor sync once the custom adapter supports online updates. They cannot establish that cached encoder outputs, self/cross KV, request `CopyState`, `CopyBatchCache`, or `CopyLogitsMixer` were rebuilt; they also cannot establish old-logp parity with the native copy-mixture policy. A policy-level fixed-batch log-probability test and a fresh-request/cache invalidation test are still required.
