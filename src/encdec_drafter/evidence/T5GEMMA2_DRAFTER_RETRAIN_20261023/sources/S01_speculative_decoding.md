# S01 — Fast Inference from Transformers via Speculative Decoding

- Authors: Yaniv Leviathan, Matan Kalman, Yossi Matias.
- Venue: ICML 2023, Proceedings of Machine Learning Research 202.
- Link: https://proceedings.mlr.press/v202/leviathan23a.html
- Source role: peer-reviewed primary paper.

The paper formalizes draft-and-verify decoding that preserves the target model's output distribution while verifying several draft tokens in parallel. It includes T5-XXL experiments, so it establishes that encoder-decoder models can use speculative decoding. It does not establish speedup for T5Gemma2, a different cache implementation, or the current drafter.
