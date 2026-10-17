# Source note: DeepSeekMath / original GRPO

- **URL:** https://arxiv.org/html/2402.03300
- **Authors / date / venue:** Shao et al., DeepSeek-AI; arXiv v3, 2024-04-27. Primary research paper; the official paper links the authors' DeepSeekMath repository.
- **Credibility:** High for the algorithm as originally proposed; it is the primary source.
- **Recency:** Medium. Foundational 2024 GRPO source, superseded in some optimizer details by 2025 analyses.
- **Bias:** Medium. The authors introduce GRPO and report favorable experiments on their math model.

## Evidence used

The method samples multiple completions from the old policy for each prompt, uses their rewards to form relative advantages, and applies a clipped token-level policy ratio. The paper's GRPO objective also divides each response's token loss by its response length and includes a KL term to the reference model. Its reported setup used 64 outputs per question, one policy update after each exploration stage, a 1e-6 learning rate, and KL coefficient 0.04. Those values are costly and math-specific; they are evidence about one recipe, not defaults for a summarizer.

The source models the task as autoregressive generation conditioned on a prompt. For an encoder-decoder, the same sequence factorization applies to decoder tokens conditioned on encoder states and the decoder prefix; that architectural transfer is mathematical inference, not an experiment reported by this paper.

> “The policy model only has a single update following each exploration stage.”

## Limitations for this decision

The experiments are decoder-only mathematical reasoning with verifiable answer rewards. They do not test encoder-decoder summarization, ROUGE rewards, or CE replay during GRPO.
