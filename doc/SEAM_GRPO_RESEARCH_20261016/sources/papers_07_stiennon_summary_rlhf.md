# Source note: RLHF summarization and an SFT reference policy

- **URL:** https://arxiv.org/html/2009.01325
- **Authors / date / venue:** Stiennon et al., OpenAI; NeurIPS 2020.
- **Credibility:** High. Peer-reviewed primary study with human comparison data, reward-model analysis, and summary-quality evaluations.
- **Recency:** Medium. Directly about RL for summarization, though it predates GRPO and uses decoder-only GPT models.
- **Bias:** Medium. The authors advocate for their human-feedback method; their reported task is English Reddit TL;DR/news.

## Evidence used

The paper trains an initially supervised summarizer with PPO against a reward model and adds a KL penalty to the SFT policy. It evaluates coverage, factual accuracy, coherence, fluency/readability, and overall quality, and finds summary length is a confound. It also warns that reference-summary training and ROUGE are proxies for summary quality. This supports using an SFT reference as a trust-region anchor and evaluating dimensions beyond ROUGE. It does not show that KL or CE is mandatory for rule-based GRPO.

> “both of these metrics are rough proxies”

## Limitations for this decision

The reward is learned from human comparisons and the optimizer is PPO, not GRPO; the architecture is decoder-only. Human-feedback and reference-based metrics are different reward regimes. The paper does not establish an encoder-decoder or Vietnamese result.
