# Controllable Summarization with Constrained Markov Decision Process

- Authors: Hou Pong Chan, Lu Wang, Irwin King.
- Venue/year: TACL 2021, volume 9, pp. 1213-1232.
- URL: https://aclanthology.org/2021.tacl-1.72/
- Inspected full text: https://aclanthology.org/2021.tacl-1.72.pdf
- Accessed: 2026-10-17.
- Type: original peer-reviewed paper.
- Credibility: high for the formulation and reported models. Recency: older foundation. Bias: author evaluation.

Verified excerpt, section 3.2: “all the Lagrange multipliers are learnable.”

The paper separates reference-similarity reward from requirement costs and updates nonnegative multipliers. Its content reward is BERTScore; its policy training uses a self-critical baseline, not GRPO. It evaluates a pointer-generator and DistilGPT2, so constraints are not exclusive to encoder-decoder models. Reusing the constrained formulation with ROUGE and SEAM GRPO is an adaptation requiring validation.
