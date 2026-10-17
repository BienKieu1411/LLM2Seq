# Length Control in Abstractive Summarization by Pretraining Information Selection

- Authors: Yizhu Liu, Qi Jia, Kenny Zhu.
- Venue/year: ACL 2022, 2022.acl-long.474.
- URL: https://aclanthology.org/2022.acl-long.474/
- Full text inspected: https://aclanthology.org/2022.acl-long.474.pdf
- Accessed: 2026-10-17, especially sections 2.2-2.3 and experiment protocol.
- Type: primary conference paper.
- Credibility: high for reported setting. Recency: established precedent. Bias: author evaluation; no SEAM replication.

Verified excerpt, section 2.2: “cross attention as a function of the desired length”.

LAAM modifies cross-attention to favor selected source positions and source EOS as decoding proceeds. Its control acts on information selection, not merely a stopping cap. It pretrains on a constructed length-balanced dataset. This supports testing budget-aware source access in SEAM; it does not justify replacing its attention or adding synthetic pretraining by default. Despite abstract wording about encoding, the concrete mechanism is in cross-attention, not a natural-language encoder prompt.
