# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

- Authors: Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, et al. (ByteDance Seed & Tsinghua AIR)
- Venue/year: arXiv preprint `arXiv:2503.14476`, 2025.
- URL: https://arxiv.org/abs/2503.14476
- Author code: https://github.com/BytedTsinghua-SIA/DAPO
- Accessed: 2026-10-17.
- Type: academic paper and official open-source RL implementation.
- Credibility: high. Recency: recent (2025). Bias: primary experiments focus on long-CoT math reasoning (AIME 2024), though Clip-Higher and dynamic zero-variance group sampling address general policy-gradient mechanics.

Verified excerpt:
> "We propose the Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) algorithm, and fully open-source a state-of-the-art large-scale RL system"

Relevance to SEAM GRPO:
1. **Clip-Higher ($\epsilon_{\text{low}}, \epsilon_{\text{high}}$):** Decouples lower and upper clipping bounds (e.g., $\epsilon_{\text{low}}=0.20, \epsilon_{\text{high}}=0.28$) so low-probability exploratory tokens with positive advantage are not immediately capped by $1+\epsilon$, mitigating entropy collapse.
2. **Dynamic Sampling for Zero-Variance Groups:** Identifies that when all $G$ completions in a group achieve identical reward, group-centered advantages are zero ($\hat{A}_i = 0$), contributing zero policy gradient while still consuming full forward/backward compute.
3. **Token-Level vs Fixed-Horizon Loss:** Contrasts token-sum normalization against sample-level normalization.
