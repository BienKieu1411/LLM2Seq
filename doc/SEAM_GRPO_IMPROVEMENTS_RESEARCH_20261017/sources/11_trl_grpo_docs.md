# Hugging Face TRL GRPOTrainer Documentation and v1.16.0 Source

- Authors: Quentin Gallouédec, Leandro von Werra, Lewis Tunstall, Kashif Rasul, et al. (Hugging Face TRL Team)
- Venue/year: Hugging Face Official Documentation & `trl==1.16.0` Release, 2025–2026.
- URL: https://huggingface.co/docs/trl/main/en/grpo_trainer
- Repository: https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_trainer.py
- Accessed: 2026-10-17.
- Type: official framework documentation and pinned library source code.
- Credibility: highest for TRL integration. Recency: current pinned version (`1.16.0`). Bias: none.

Verified excerpt:
> "Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but waking the engine adds host–device transfer latency."

Relevance to SEAM GRPO:
Documents TRL's native support for `epsilon_high` (asymmetric clipping), `loss_type` (`"grpo"`, `"bnpo"`, `"dr_grpo"`, `"dapo"`), `scale_rewards` (`"group"`, `"batch"`, `"none"`), `mask_truncated_completions`, and `importance_sampling_level` (`"token"`, `"sequence"`).
