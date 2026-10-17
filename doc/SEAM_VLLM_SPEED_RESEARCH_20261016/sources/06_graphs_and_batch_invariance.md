# S6-S7: CUDA graphs and batch invariance

- **Sources:** [PyTorch 2.10 CUDA semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html), [vLLM 0.25.1 CUDA graph modes](https://docs.vllm.ai/en/v0.25.1/design/cuda_graphs/), [vLLM 0.25.1 batch invariance](https://docs.vllm.ai/en/v0.25.1/features/batch_invariance/)
- **Type / credibility:** Official PyTorch and version-pinned vLLM docs.
- **Relevant evidence:** PyTorch graphs replay the same captured addresses and require static shapes/control flow. vLLM provides `FULL_DECODE_ONLY`, which captures uniform decode while leaving prefill/mixed batches eager. vLLM's batch-invariance mode targets outputs independent of batch size/order but is marked beta and may lower performance.
- **Application:** A decode-only graph is a plausible future speed experiment. It is not safe to enable in this adapter by changing a CLI flag alone: `SEAMForQwen3.forward` updates Python-side pending copy state, while graph replay does not rerun ordinary Python setup in the captured model path. The request-specific state must first be staged outside replay and tested under request churn. Batch invariance could help compare outputs across concurrency values but is not an optimization for speed by itself.
- **Limit:** No graph or batch-invariance GPU experiment was run; these docs do not establish the feature works with SEAM's custom vLLM state class.
