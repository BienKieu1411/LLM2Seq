# S1: PyTorch out buffers and elementwise operations

- **Source:** [torch.sub](https://docs.pytorch.org/docs/2.10/generated/torch.sub.html), [torch.empty_like](https://docs.pytorch.org/docs/2.10/generated/torch.empty_like.html), [torch.where](https://docs.pytorch.org/docs/2.10/generated/torch.where.html)
- **Type / credibility:** Official PyTorch 2.10 API documentation; primary source for public API behavior.
- **Relevant text:** `torch.sub` accepts `out=` and supports broadcasting/type promotion. `torch.where` also accepts `out=` and selects each element from one of two inputs. `empty_like` allocates an uninitialized output with a matching shape.
- **Application:** The custom copy mixer can subtract into a dedicated FP32 output buffer, add the generation term in place, and select the generation-only fallback into that same buffer. These documented APIs make the one-buffer implementation possible.
- **Limit:** The docs do not prove that the SEAM result equals the old result or that it is faster on the server. Exact equality is checked against the local AFMR reference; target-GPU parity and timing remain outstanding.
