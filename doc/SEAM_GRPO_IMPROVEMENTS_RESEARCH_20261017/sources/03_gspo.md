# Group Sequence Policy Optimization (GSPO)

- Authors: Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, Junyang Lin (Qwen Team, Alibaba Inc.)
- Venue/year: arXiv preprint `arXiv:2507.18071`, July 2025.
- URL: https://arxiv.org/abs/2507.18071
- Accessed: 2026-10-17.
- Type: academic paper and algorithm behind Qwen3 RL post-training; supported in TRL 1.16.0 via `importance_sampling_level="sequence"`.
- Credibility: high. Recency: recent (July 2025). Bias: author evaluation on Qwen models.

Verified excerpt:
> "GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization."

Relevance to SEAM GRPO:
Standard GRPO computes token-level importance ratios $r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x, y_{i,<t})}$ even though rewards (ROUGE, length error, source discrimination) are sequence-level. GSPO uses the geometric-mean sequence ratio $s_i(\theta) = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\right)$, reducing high-variance single-token ratio spikes (especially relevant when a copy-gate shifts probabilities on specific named-entity tokens or when `num_iterations > 1`).
