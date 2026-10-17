# MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention (CISPO)

- Authors: Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, et al. (MiniMax)
- Venue/year: arXiv preprint `arXiv:2506.13585`, June 2025.
- URL: https://arxiv.org/abs/2506.13585
- Accessed: 2026-10-17.
- Type: technical report / academic paper.
- Credibility: high. Recency: recent (June 2025). Bias: author evaluation.

Verified excerpt:
> "we propose CISPO, a novel RL algorithm to further enhance RL efficiency. CISPO clips importance sampling weights rather than token updates"

Relevance to SEAM GRPO:
When PPO/GRPO clips $\min(r_{i,t} A_i, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) A_i)$, tokens that cross the clip boundary receive zero gradient. CISPO instead clips the detached importance weight $\text{sg}(\text{clip}(r_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}))$ multiplying $\log \pi_\theta$, preserving gradient flow on rare pivot tokens during multi-step updates.
