# SEAM GRPO Training Improvements: Deep Research Plan

Date: 2026-10-17
Status: Active Investigation (`deep-research` pipeline)
Target systems: `src/seam_grpo`, `src/seam_length_grpo`, and `src/eviseq_new/seam_vllm`

## Phase 1: Reframe & Falsifiable Hypotheses

### Underlying Decision
Determine which concrete, high-leverage improvements (across **systems/compute efficiency**, **RL objective/advantage estimation**, and **multi-objective reward/length-content coupling**) should be prioritized for `seam_grpo` and `seam_length_grpo` without breaking the verified full-actor conditional SEAM architecture contract.

### Falsifiable Hypotheses

- **H1 (Group-Level Unique-Source Encoding & Collation):** Because all $G$ candidates in a GRPO group share the exact same `(source, target_words, encoder_prompt, decoder_prompt, output_budget, policy_version)`, deduplicating collation (`LengthCollator`), teacher source encoding (`SourceTeacher`), native rollout encoding (`rollout.py`), and differentiable `encode_source` (`Policy.forward` / `SEAMActor._score_candidates`) from $O(B \cdot G)$ to $O(B)$ unique sources per group will reduce encoder/bridge/copy-preparation compute and activation memory by a factor of $G$ (e.g., $4\times$ at $G=4$) while producing **bitwise or numerically identical** log-probabilities and parameter gradients.
- **H2 (Decoupled / Variance-Aware Multi-Objective Advantage vs Linear Scalarization in `seam_length_grpo`):** In linear scalarization $R_i = Q_i + \eta S_{\text{src},i} - \lambda C_{\text{len},i}$, group centering $A_i = R_i - \bar{R}_g$ allows a high-variance penalty (such as `length_cost` when one candidate is truncated or short) to dominate the group advantage sign and magnitude even with `scale_rewards="none"`. Group-wise component normalization / decoupled advantage combination or dual/gated constraints will preserve ROUGE signal better than raw linear subtraction when component variances differ by $>3\times$.
- **H3 (Asymmetric Clipping `epsilon_high` & Zero-Gradient Group Filtering in TRL 1.16.0):** Pinned `trl==1.16.0` natively supports asymmetric PPO/GRPO clipping (`epsilon_high`, DAPO Clip-Higher) and entropy/clip diagnostics without new custom loss code; enabling `epsilon_high > epsilon_low` (e.g., $0.28$ vs $0.20$) and skipping backward passes on zero-advantage groups (`all(|A_i| < eps)` when `ce_weight == 0`) or applying dynamic resampling reduces wasted updates and mitigates premature entropy collapse.
- **H4 (Memory & Throughput Optimization for Colocated vLLM + Frozen Teacher):** Reusing the frozen SFT model's true-source forward pass between KL reference (`beta > 0`) and `SourceTeacher` (when `temperature == 1.0`), or enabling vLLM `n=G` per unique source request (instead of $G$ duplicate `n=1` requests that re-run the multimodal processor and encoder $G$ times), will cut rollout and teacher overhead significantly without violating cache invalidation across `policy_version` updates.

## Phase 2: Genre & Building Blocks

- **Genre:** `decision` + `validation` (Technical Architecture & Algorithmic Improvement Study)
- **Building Blocks:**
  1. Executive Decision Matrix (Impact vs Engineering Risk vs Architecture Compatibility)
  2. Systems & Compute Bottleneck Audit (Exact FLOP/Memory/Call-count trace in `seam_grpo` and `seam_length_grpo`)
  3. Algorithmic GRPO Objective & Advantage Audit (Dr. GRPO, DAPO, GSPO, Multi-objective GRPO / GR3 / GDPO)
  4. Triangulated Evidence Table & Hypothesis Verdicts (H1–H4)
  5. Adversarial Self-Critique & Counter-Arguments (Why some popular RL tricks fail on conditional copy-mixture encoder-decoders)
  6. Phased Implementation Roadmap & Verification Protocol

## Phase 3 & 3.5: Sourcing Strategy, Capability Discovery & Risk Register

- **Channels:**
  1. **Local Codebase & Runtime Inspection:** Exact line-by-line audit of `src/seam_grpo`, `src/seam_length_grpo`, `src/eviseq_new`, and installed `trl==1.16.0` / `vllm` interfaces in `/Users/kieugiangbien/bienkieu_env`.
  2. **Primary Academic Literature (2024–2026):** DeepSeekMath (GRPO), Dr. GRPO, DAPO, GSPO (Group Sequence Policy Optimization), GR3 (ICML 2026), Multi-Objective GRPO / GDPO, CMDP Summarization, Paulus et al. / ROUGESal.
  3. **Official Framework & Production RL Codebases:** HuggingFace TRL 1.16.0 source, VeRL / Slime / OpenRLHF / vLLM V1 encoder-decoder & sleep-mode documentation.
- **Opposition / Adversarial Queries:**
  - When does `n=G` in vLLM encoder-decoder fail or diverge from `n=1`?
  - Does sharing `encode_source` across $G$ candidates in PyTorch autograd change gradient accumulation or DDP bucket synchronization?
  - When does Clip-Higher (`epsilon_high`) or multi-epoch reuse (`num_iterations > 1`) cause reward hacking or copy-gate collapse in summarization?
- **Stop Criteria:** Every hypothesis H1–H4 is backed by $\ge 3$ independent sources (primary paper, framework/production code, and local codebase verification/math check) and stress-tested in the adversarial pass.
