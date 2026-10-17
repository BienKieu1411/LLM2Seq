# Mutual Information Alleviates Hallucinations in Abstractive Summarization

- Authors: Liam van der Poel, Ryan Cotterell, Clara Meister.
- Venue/year: EMNLP 2022, 2022.emnlp-main.399.
- URL: https://aclanthology.org/2022.emnlp-main.399/
- Full text inspected: https://aclanthology.org/2022.emnlp-main.399.pdf
- Accessed: 2026-10-17.
- Type: primary conference paper.
- Credibility: high within reported scope. Recency: established precedent. Bias: author evaluation.

Verified excerpt, abstract: “pointwise mutual information of the source and target token”.

The method contrasts source-conditional token likelihood with a target-only language model at inference, gated by uncertainty. It motivates examining source dependence. A frozen SEAM teacher contrasted against a matched wrong source is a different construction: it is not this paper's CPMI, a mutual-information estimate, or a demonstrated GRPO reward. It adds computation and can carry teacher bias.
