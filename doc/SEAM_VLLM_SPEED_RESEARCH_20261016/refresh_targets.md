# Refresh targets

- Recheck the vLLM and PyTorch source links if the server moves away from vLLM 0.25.1 / PyTorch 2.12.
- Run CUDA parity for copy logits with FP32/BF16, strided/padded logits, all-invalid rows, and request reorder.
- Compare predictions and ROUGE-1.5.5 using fresh output directories at fixed request order and sampling configuration.
- Sweep `max_num_seqs` and `encoder_max_batch_tokens`; record examples/s, output tokens/s, peak VRAM/KV blocks, and output equality.
- Only then decide whether decode-only CUDA graphs or length-aware encoder bucketing are worth implementing.
