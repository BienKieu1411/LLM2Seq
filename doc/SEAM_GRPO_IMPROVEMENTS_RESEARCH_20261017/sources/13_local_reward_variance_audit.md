# Local Code & Numerical Audit: LengthReward Variance Dominance and Flat Groups

- Source: `src/seam_length_grpo/seam_length_grpo/reward.py:28-50`, `src/seam_length_grpo/seam_length_grpo/trainer.py:120-124`, `src/seam_length_grpo/configs/grpo.yaml:14,38-57`, `src/seam_grpo/seam_grpo/config.py:21`.
- Accessed: 2026-10-17.
- Type: primary local codebase and mathematical/numerical verification.
- Credibility: direct code inspection and exact arithmetic. Recency: current workspace snapshot. Bias: none.

Verified code excerpt (`src/seam_length_grpo/seam_length_grpo/reward.py:38-43`):
```python
error = abs(actual - target_words)
cost = error / error_budget
return {
    **scores,
    "quality": quality,
    "reward": quality + source_weight * source_score - self.length_weight * cost,
```

Key findings:
1. **Unnormalized group centering (`scale_rewards="none"`):** Advantage is $A_i = (Q_i - \bar{Q}) + \eta(S_i - \bar{S}) - \lambda(C_i - \bar{C})$. Because $Q_i \in [0, 1]$ has small intra-group spread ($\Delta Q \approx 0.01\text{--}0.04$ across $G=4$ rollouts from SFT) while $\lambda C_i = \frac{0.10}{B_d}|L_i - W^*|$ is unbounded above ($0.00294$/word on `vlsp22` with $B_d=34$, $0.00263$/word on `vietnews` with $B_d=38$), a word-error difference of just 11–12 words completely reverses a $+3.0$ ROUGE F1 point advantage.
2. **Dataset-dependent gradient scale:** A 20-word spread on `vlsp22` ($B_d=34$) produces a $2.88\times$ larger advantage magnitude than the same 20-word spread on `ViM` ($B_d=98$).
3. **Outlier candidate distortion:** A single verbose candidate (e.g., 170 words over target) pulls down the group mean $\bar{R}$ and assigns large positive advantages ($+0.11$ to $+0.17$) to all other candidates in the group, even if their ROUGE $Q_i$ is below average.
4. **No short-circuiting on zero-variance / tied groups:** When all $G=4$ candidates have identical rewards ($A_i = 0$), `SEAMActor.forward` still runs $G$ full encoder+bridge+decoder backward passes and `SourceTeacher` runs $2G$ forward passes.
