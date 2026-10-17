# Your Efficient RL Framework Secretly Brings You Off-Policy RL Training (TIS)

- Authors: Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, Jianfeng Gao (UCSD, Microsoft Research, Mila)
- Venue/year: Technical Report, August 2025.
- URL: https://fengyao.notion.site/off-policy-rl
- Accessed: 2026-10-17.
- Type: academic technical report and production RL analysis.
- Credibility: high. Recency: recent (August 2025). Bias: low (identifies subtle numerical mismatch between inference and training kernels).

Verified excerpt:
> "The policy used to generate rollouts is different from the policy being trained due to numerical discrepancies between inference and training engines."

Relevance to SEAM GRPO:
Even when weights are freshly synchronized (`policy_version=k`, `num_iterations=1`), vLLM's paged cross-attention / FlashAttention kernels and PyTorch SDPA training kernels produce small numerical differences in $\log \pi$. Keeping `old_per_token_logps = sampling_per_token_logps` (as `SEAMGRPOTrainer` does) paired with a bounded importance ratio / parity guard (`vllm_logp_mean_tolerance`, `vllm_logp_max_tolerance`) directly addresses this off-policy kernel discrepancy.
