# S5: Heterogeneous `foreach_copy_` behavior

- **Source:** [PyTorch 2.12 CUDA foreach copy implementation](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/cuda/ForeachBinaryOpList.cu)
- **Type / credibility:** Version-pinned upstream CUDA implementation; primary source for this kernel dispatch.
- **Relevant code:** The fast path checks that tensor lists share sizes and strides; otherwise it calls `foreach_tensor_copy_list_kernel_slow_`.
- **Application:** SEAM's source-copy rows usually have different widths, so the current `torch._foreach_copy_` call can reduce Python dispatch but should not be described as fusing all CUDA copies.
- **Limit:** This is PyTorch 2.12 source. The server's exact PyTorch version should be checked before extrapolating implementation details.
