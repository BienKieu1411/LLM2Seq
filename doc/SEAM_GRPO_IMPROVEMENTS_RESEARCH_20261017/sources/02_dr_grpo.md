# Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)

- Authors: Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin (Sea AI Lab, NUS, SMU)
- Venue/year: COLM 2025 / `arXiv:2503.20783`, 2025.
- URL: https://arxiv.org/abs/2503.20783
- Author code: https://github.com/sail-sg/understand-r1-zero
- Accessed: 2026-10-17.
- Type: peer-reviewed conference paper and official repository.
- Credibility: very high. Recency: recent (2025). Bias: low (critiques optimization biases in standard GRPO).

Verified excerpt:
> "To address this, we introduce Dr. GRPO, an unbiased optimization method that improves token efficiency while maintaining reasoning performance."

Relevance to SEAM GRPO:
Proves mathematically that dividing per-trajectory token losses by realized length $|o_i|$ biases negative-advantage updates toward longer responses and positive-advantage updates toward shorter responses, and that dividing by per-group reward standard deviation $\text{std}(R_1, \dots, R_G)$ over-weights low-variance groups. Both `seam_grpo` and `seam_length_grpo` already adopt `loss_type="dr_grpo"` and `scale_rewards="none"` as their default baseline.
