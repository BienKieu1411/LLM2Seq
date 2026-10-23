# S04 — Acceptance-Aware Draft Model Training for Speculative Decoding

- Authors: Tianhua Xia, Mugilan Ganesan, Yifei Feng, Haiyu Wang, Maximilian Egger, Sai Qian Zhang.
- Link: https://arxiv.org/abs/2609.24150
- Source role: recent primary preprint.

The paper distinguishes greedy acceptance from sampling acceptance. For greedy verification, a token is accepted when the draft proposes the target argmax; its expected accepted-length objective sums cumulative products of the draft probability assigned to target-greedy tokens over a speculative window. The paper separately proposes window total variation for sampling and a further GRPO stage.

The repository's runtime proposes draft tokens with deterministic `argmax`, so its exact greedy acceptance indicator is whether the draft argmax equals the target argmax. The preprint's probability-mass EAL instead uses the draft probability assigned to the target-greedy token; that quantity describes acceptance mass for stochastic proposals and is not the exact objective for this deterministic runtime. The implementation therefore uses a sigmoid of the target-token-vs-best-competitor logit margin as a differentiable top-1 surrogate, multiplies these surrogates across a prefix, and adds a small CE term after warm-up. The surrogate is not calibrated as an acceptance probability. This is an engineering adaptation, not a reproduction of the preprint's objective, data, training schedule, or reported result.
