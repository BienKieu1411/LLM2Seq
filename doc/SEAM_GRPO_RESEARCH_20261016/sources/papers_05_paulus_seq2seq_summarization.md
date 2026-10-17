# Source note: mixed RL and supervised loss for encoder-decoder summarization

- **URL:** https://arxiv.org/html/1705.04304
- **Authors / date / venue:** Paulus, Xiong, and Socher; arXiv v3, 2017-11-13. Primary summarization research paper.
- **Credibility:** High for its reported encoder-decoder summarization experiment; source is older and predates Transformer models.
- **Recency:** Low for present-day model engineering, but still directly relevant to the sequence-to-sequence loss design.
- **Bias:** Medium. The authors propose and evaluate a hybrid objective.

## Evidence used

This work directly trains an encoder-decoder abstractive summarizer with policy-gradient reward and maximum-likelihood cross-entropy. Its mixed objective is motivated by readability and metric gaming. On CNN/DailyMail, pure RL obtained higher ROUGE than the mixed objective, but was less readable; the mixed objective scored better than MLE on ROUGE and received better human readability/relevance ratings than pure RL. The authors also report that ROUGE-2 rewards caused summaries to hit the maximum length and end abruptly. This is strong evidence for testing an SFT/CE anchor and monitoring length, but it does not quantify an optimal CE coefficient for modern GRPO.

> “combines standard supervised word prediction and reinforcement learning”

## Limitations for this decision

The model is recurrent and the RL optimizer is self-critical policy gradient, not GRPO. Its English CNN/DailyMail and NYT results do not establish outcomes for the project's custom architecture or Vietnamese evaluations.
