# GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization

- Authors: Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, Pavlo Molchanov (NVIDIA & HKUST)
- Venue/year: arXiv preprint `arXiv:2601.05242`, January 2026.
- URL: https://arxiv.org/abs/2601.05242
- Author code: https://github.com/NVlabs/GDPO
- Accessed: 2026-10-17.
- Type: academic paper and official NVIDIA repository.
- Credibility: very high. Recency: cutting-edge (January 2026). Bias: author evaluation across multi-reward tasks (reasoning + length/format constraints).

Verified excerpt:
> "GDPO... resolve[s] these issues by decoupling the normalization of individual rewards, more faithfully preserving their relative differences and enabling more accurate multi-reward optimization"

Relevance to SEAM GRPO (`seam_length_grpo`):
In `seam_length_grpo`, three rewards with very different scales and intra-group variances are combined linearly before group centering:
$R_i = Q_i + \eta S_{\text{src}, i} - \lambda C_{\text{len}, i}$.
Because `C_len = |words - W| / B` can vary by $0.5\text{--}2.0$ across candidates while ROUGE F1 $Q_i$ often varies by only $0.01\text{--}0.04$ within a group of $G=4$ samples from the same SFT policy, the centered advantage $A_i = R_i - \bar{R}_g$ is frequently dominated by length error, washing out the ROUGE preference signal. GDPO shows that normalizing/centering each reward component separately before weighted combination preserves the relative preference signal of low-variance quality metrics alongside length/format constraints.
