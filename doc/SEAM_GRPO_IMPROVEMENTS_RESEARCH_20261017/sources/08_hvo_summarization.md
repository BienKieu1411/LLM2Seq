# Balancing Rewards in Text Summarization: Multi-Objective Reinforcement Learning via HyperVolume Optimization (HVO)

- Authors: Junjie Song, Yiwen Liu, Dapeng Li, Yin Sun, Shukun Fu, Siqi Chen, Yuji Cao
- Venue/year: arXiv preprint `arXiv:2510.19325`, October 2025.
- URL: https://arxiv.org/abs/2510.19325
- Accessed: 2026-10-17.
- Type: academic preprint.
- Credibility: moderate-to-high (provisional preprint, directly on multi-reward summarization with GRPO). Recency: recent (October 2025). Bias: author evaluation.

Verified excerpt:
> "we introduce hypervolume optimization (HVO), a novel optimization strategy that dynamically adjusts the scores between groups during the reward process in RL"

Relevance to SEAM GRPO:
Demonstrates that static linear weights across summarization dimensions (ROUGE/relevance, consistency, length) suffer from dimension collapse when one metric is easier to optimize than another, motivating Pareto/variance-balanced multi-objective aggregation.
