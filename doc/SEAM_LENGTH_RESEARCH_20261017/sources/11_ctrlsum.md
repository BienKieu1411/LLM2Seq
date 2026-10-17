# CTRLsum: Towards Generic Controllable Text Summarization

- Authors: Junxian He, Wojciech Kryscinski, Bryan McCann, Nazneen Rajani, Caiming Xiong.
- Venue/year: EMNLP 2022, 2022.emnlp-main.396.
- URL: https://aclanthology.org/2022.emnlp-main.396/
- Full text inspected: https://aclanthology.org/2022.emnlp-main.396.pdf
- Accessed: 2026-10-17.
- Type: primary conference paper.
- Credibility: high within reported scope. Recency: established precedent. Bias: author experiments, no SEAM replication.

Verified excerpt, section 3.2: “keywords help focus on control-related content on the encoder side”.

The paper combines encoder-side keyword guidance with a decoder prefix. It tests the combination against individual controls for contribution/purpose tasks. Its length mechanism instead maps length bins to keyword counts; it does not establish exact-word control through two natural-language prompts. Take the separation of content guidance and output control, not a claim of proven SEAM length gains.
