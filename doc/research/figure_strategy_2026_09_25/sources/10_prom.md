# Ma et al. (2024): phrase copying in summarization

- Primary record: https://aclanthology.org/2024.lrec-main.1148/
- Type: original summarization paper; accessed 2026-09-25.
- Credibility 5/5; recency 4/5; low-bias 3/5 (authors report their own method).
- Short quote from abstract: “tokens in n-gram that can be copied from the source”.
- Relevant observation: copying can be evaluated explicitly rather than inferred from ROUGE. A SEAM plot should measure a targeted copy outcome, not simply repeat the overall score table.
- Limit: PROM has pretraining and an auxiliary copying objective, unlike SEAM's single likelihood objective. The prior local audit records small or mixed long-document gains; the result is no performance forecast for SEAM.

