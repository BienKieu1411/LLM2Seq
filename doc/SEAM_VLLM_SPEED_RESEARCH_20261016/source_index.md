# Source index

| ID | Source | Type | Used for | Limits |
|---|---|---|---|---|
| S1 | [PyTorch 2.10 `torch.sub`](https://docs.pytorch.org/docs/2.10/generated/torch.sub.html), [`empty_like`](https://docs.pytorch.org/docs/2.10/generated/torch.empty_like.html), [`where`](https://docs.pytorch.org/docs/2.10/generated/torch.where.html) | Official API docs | Out buffers, broadcasting, selected-value semantics | Documents API behavior, not GPU speed or numerical equivalence of this adapter |
| S2 | [PyTorch 2.10 `Tensor.tolist`](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.tolist.html) | Official API docs | Device-to-host copy in encoder length extraction | Does not quantify sync latency |
| S3 | [vLLM 0.25.1 `EncoderDecoderModelState`](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu/model_states/encoder_decoder.py) | Version-pinned upstream implementation | Encoder output lifetime and cross-attention KV reuse | General vLLM implementation; SEAM request state was inspected separately |
| S4 | [vLLM 0.25.1 memory guide](https://docs.vllm.ai/en/v0.25.1/configuration/conserving_memory/) | Version-pinned official docs | `max_num_seqs` and memory trade-offs | Does not prescribe best throughput for SEAM |
| S5 | [PyTorch 2.12 CUDA foreach copy implementation](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/cuda/ForeachBinaryOpList.cu) | Version-pinned upstream implementation | Heterogeneous tensor copies may take the slower list path | PyTorch 2.12 source, not every server version |
| S6 | [PyTorch CUDA graph constraints](https://docs.pytorch.org/docs/2.10/notes/cuda.html), [vLLM 0.25.1 CUDA graph modes](https://docs.vllm.ai/en/v0.25.1/design/cuda_graphs/) | Official framework docs | Why decode-only graphs are plausible but not a config-only safe switch | Must stage SEAM's Python copy state and validate on target GPU |
| S7 | [vLLM 0.25.1 batch invariance](https://docs.vllm.ai/en/v0.25.1/features/batch_invariance/) | Version-pinned official docs | Exact-output reproducibility across batch schedules; possible speed cost | Beta feature; no SEAM compatibility or performance benchmark |
| S8 | Local implementation and reference tests in `src/eviseq_new` | Project evidence | Actual tensor layouts, state routing, CPU exactness, test results | CPU results cannot establish CUDA parity or throughput |
