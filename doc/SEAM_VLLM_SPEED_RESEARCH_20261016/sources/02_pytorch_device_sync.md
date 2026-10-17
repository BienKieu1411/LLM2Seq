# S2: Device-to-host length extraction

- **Source:** [PyTorch 2.10 `Tensor.tolist`](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.tolist.html)
- **Type / credibility:** Official PyTorch API documentation; primary source.
- **Relevant text:** “Tensors are automatically moved to the CPU first if necessary.”
- **Application:** `attention_mask.sum(-1).tolist()` in the SEAM encoder adapter requires reading GPU-derived values on the host. Moving it outside the microbatch loop reduces the number of these host-visible reads to one per incoming encoder batch.
- **Limit:** The documentation confirms the device transfer, not the duration or whether it dominates a production request. The speed effect must be measured on the actual GPU.
