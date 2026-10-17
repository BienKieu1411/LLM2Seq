# Leash: Adaptive Length Penalty and Reward Shaping for Efficient Large Reasoning Model

- Authors: Yanhao Li, Lu Ma, Jiaran Zhang, Lexiang Tang, Wentao Zhang, Guibo Luo (Peking University)
- Venue/year: arXiv preprint `arXiv:2512.21540`, December 2025.
- URL: https://arxiv.org/abs/2512.21540
- Accessed: 2026-10-17.
- Type: academic preprint; extends CMDP primal-dual length control (Kikuchi/Paulus/TACL 2021 CMDP) to GRPO.
- Credibility: high. Recency: recent (December 2025). Bias: author evaluation.

Verified excerpt:
> "We formulate length control as a constrained optimization problem and employ a Lagrangian primal–dual method to dynamically adjust the penalty coefficient."

Relevance to SEAM GRPO (`seam_length_grpo`):
In `seam_length_grpo`, `length_weight=0.10` is fixed across all 7 datasets, even though datasets like WikiLingua and XLSum already satisfy their MAE budgets ($\text{cost} \le 1.0$) while Vietnews or ViM violate theirs. A per-dataset dual multiplier $\lambda_d \leftarrow \text{clip}(\lambda_d + \eta_\lambda (\text{mean\_cost}_d - 1.0), 0, \lambda_{\max})$ updated at optimizer step boundaries (using global reduced dataset costs) automatically relaxes the length penalty on datasets that already meet their MAE target so GRPO can focus 100% on ROUGE $Q$, while tightening $\lambda_d$ only on datasets that violate their length budget.
