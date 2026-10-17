# vLLM V1 EncoderCacheManager & Multimodal Encoder Caching

- Authors: Woosuk Kwon, Zhuohan Li, Simon Mo, Roger Wang, Kaichao You, et al. (vLLM Project Team)
- Venue/year: vLLM Official Documentation & Source Code (`vllm/v1/core/encoder_cache_manager.py`), 2025–2026.
- URL: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/encoder_cache_manager.py
- Accessed: 2026-10-17.
- Type: primary framework source code.
- Credibility: highest for vLLM internals. Recency: current (vLLM V1 / 0.25.1). Bias: none.

Verified excerpt:
> "Manages caching of encoder outputs for multimodal models in vLLM V1. The EncoderCacheManager handles the lifecycle of multimodal encoder outputs... during request processing."

Relevance to SEAM GRPO:
Explains how vLLM V1 shares encoded source/multimodal representations across multiple candidate sequences within the same policy version, whereas repeating $G$ separate `n=1` requests with `mm_processor_cache_gb=0` forces full re-tokenization, copy alignment, and encoder+bridge execution $G$ times per prompt.
