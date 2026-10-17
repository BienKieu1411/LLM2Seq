# Source note: GRPO objective and novelty boundary

**Source:** Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), arXiv:2402.03300, 2024. Primary paper; accessed 2026-10-17.

**Short quote:** “we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO)” (abstract, 17 words).

**Relevant evidence:** The paper introduces the group-relative policy-optimization objective based on responses sampled from an old policy, group-relative advantages, and a clipped policy ratio. TRL's current docs provide the implemented token-level ratio and normalization variants.

**Application to SEAM:** The model's conditional distribution may be encoder-decoder and copy-mixture; the policy ratio still compares the current probability of each sampled action with its behavior probability. Architecture changes the probability computation and synchronization scope. It is not, by itself, a new GRPO estimator/objective.

**Limit:** The original paper does not describe SEAM or its encoder/bridge/copy distribution. The conclusion is a boundary between objective mechanics and model integration, not an evaluation of research novelty of a future modified objective.
