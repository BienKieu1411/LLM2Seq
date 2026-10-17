# S4: vLLM concurrency and memory

- **Source:** [vLLM 0.25.1 Conserving Memory](https://docs.vllm.ai/en/v0.25.1/configuration/conserving_memory/)
- **Type / credibility:** Official version-pinned vLLM documentation.
- **Relevant text:** The guide identifies `max_model_len` and `max_num_seqs` as controls for memory use.
- **Application:** The user's OOM log had 249 active requests, 224 MiB free, and then failed on an additional 488 MiB request. Testing a lower sequence concurrency while preserving the exact input/output token limits is a reasonable memory/throughput sweep.
- **Limit:** Lower concurrency may reduce throughput on some workloads; the docs do not provide a SEAM-specific optimum. It must be selected using throughput and peak-memory measurements on the target GPU.
