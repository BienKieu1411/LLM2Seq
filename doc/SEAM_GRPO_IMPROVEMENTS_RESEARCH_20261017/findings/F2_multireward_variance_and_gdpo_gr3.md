# Finding F2 — Multi-Reward Variance Dominance in `seam_length_grpo` and Decoupled / Multiplicative Fixes (GDPO & GR3)

- **Thesis:** Under linear scalarization $R_i = Q_i + \eta S_i - \lambda C_i$ with `scale_rewards="none"`, unbounded length deviation cost $C_i = |L_i - W^*|/B_d$ dominates bounded ROUGE quality $Q_i \in [0, 1]$ whenever a group contains a verbose outlier or tight bucket width $B_d \in \{34, 38\}$ (`vlsp22`, `vietnews`), allowing an 11.5-word length delta to overturn a $+3.0$ ROUGE-F1 point advantage. Decoupled group normalization (**GDPO**, `arXiv:2601.05242`) or non-negative quality rescaling with multiplicative length decay (**GR3**, `ICML 2026`, `arXiv:2603.10535`), paired with a Huber-clipped $C_i \le C_{\max}$, eliminates length-outlier hijacking.
- **Status:** Confirmed (Triangulated across Sources [02](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/02_dr_grpo.md), [06](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/06_gdpo_multireward.md), [07](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/07_gr3_icml2026.md), [09](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/09_leash_cmdp.md), [13](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/13_local_reward_variance_audit.md)).

## Evidence & Quantitative Verification

1. **Trade-off Ratio (`src/seam_length_grpo/seam_length_grpo/reward.py:92-119`, `grpo.yaml`):**
   - With `length_weight` $\lambda = 0.10$ and `vlsp22` bucket width $B_d = 34$, each 1-word deviation costs $0.10 / 34 = 0.002941$ reward units.
   - Within a group of $G=4$ candidates from the same policy, weighted ROUGE F1 (`rouge1: 0.20, rouge2: 0.50, rougeL: 0.30`) typically varies by $\Delta Q \in [0.015, 0.035]$.
   - Consequently, $\Delta|L - W^*| \ge 12$ words completely erases a $+0.035$ ($+3.5$ ROUGE-F1 point) quality advantage.

2. **Verbose Outlier Hijacking:**
   - Because $A_i = (Q_i - \bar{Q}_g) - \lambda(C_i - \bar{C}_g)$, a single 250-word runaway candidate ($C_4 = 5.0$, $\lambda C_4 = 0.50$) inflates $\lambda \bar{C}_g = 0.145$, handing all three remaining candidates large positive advantages ($+0.11$ to $+0.17$) almost entirely independent of their ROUGE differences.

3. **Three Complementary Fixes:**
   - **Fix 2A (Bounded / Clipped Length Cost):** Clamp $C_i = \min(C_{\max}, |L_i - W^*|/B_d)$ (e.g., $C_{\max} = 2.0$ or smooth $\tanh(|L_i - W^*| / (2 B_d))$) so a single runaway generation cannot skew the group mean $\bar{C}_g$ for the other $G-1$ candidates.
   - **Fix 2B (GDPO / Decoupled Group Advantage or TRL `multi_objective_aggregation="normalize_then_sum"`):** Normalize each reward component within the group (with an $\epsilon + \sigma_{\min}$ floor so near-zero variance within a group does not amplify noise) before combining with weights $(1.0, \eta, \lambda)$.
   - **Fix 2C (GR3 Multiplicative Gating for Length Compliance):** Shift quality within the group $\tilde{Q}_i = Q_i - \min_{j \in g} Q_j + \epsilon_0 \ge 0$ and scale multiplicatively by length compliance $w_{\text{len}}(C_i) = \exp(-\alpha \max(0, C_i - \tau_{\text{tol}}))$, ensuring a low-ROUGE summary can **never** receive a positive advantage solely for hitting the target word count.
