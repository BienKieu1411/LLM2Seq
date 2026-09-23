# S05 — Convolution for Large Language Models, technical report

URL: https://arxiv.org/html/2607.18413v1
Authors: Tian et al. Date: 2026-07-20. Accessed: 2026-09-23.
Type: primary preprint. Credibility 3/5; recency 5/5; advocacy risk medium.
Verified excerpt, section 3.5: “None improves both loss and perplexity over the selected configuration.”

Qwen3 experiments compare convolution placement and favor post-QKV residual
depthwise convolution, width three. Pre-QKV placement also improves reported
perplexity, but some positions hurt; additional normalization/activation or
multiple convolution branches are not uniformly beneficial. Initialization is
also sensitive. This is from-scratch language-model pretraining, not bridge
fine-tuning. Neither our zero-DC constraint nor key-only application is tested.
