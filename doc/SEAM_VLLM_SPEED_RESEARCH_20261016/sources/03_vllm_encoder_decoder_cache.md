# S3: vLLM encoder-decoder cache lifetime

- **Source:** [vLLM 0.25.1 `EncoderDecoderModelState`](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu/model_states/encoder_decoder.py)
- **Type / credibility:** Version-pinned upstream implementation; primary source for the exact server version in use.
- **Relevant text:** The source labels the decode path “Decode steps: encoder K/V are in cross-attention KV cache.” It clears `encoder_outputs` after handing them to the decoder.
- **Application:** SEAM's `memory_norm` and cross-K/V projection populate cached encoder K/V during prefill. Recomputing/caching normalized cross-memory inside every token decode is not a useful target: that work is not repeated on decode steps.
- **Limit:** This establishes vLLM 0.25.1's general encoder-decoder flow. SEAM's custom key/value path and request alignment were separately audited in `seam_vllm/model.py` and `seam_vllm/state.py`.
