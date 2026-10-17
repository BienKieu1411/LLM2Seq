# GSum: A General Framework for Guided Neural Abstractive Summarization

- Authors: Zi-Yi Dou, Pengfei Liu, Hiroaki Hayashi, Zhengbao Jiang, Graham Neubig.
- Venue/year: NAACL 2021, 2021.naacl-main.384.
- URL: https://aclanthology.org/2021.naacl-main.384/
- Full text inspected: https://arxiv.org/pdf/2010.08014
- Accessed: 2026-10-17.
- Type: primary conference paper / author preprint full text.
- Credibility: high within scope. Recency: established precedent. Bias: author experiments.

Verified excerpt, abstract: “different kinds of external guidance as input”.

GSum encodes source and guidance, with decoder attention to each. Its extra guidance encoder and attention path are not SEAM's existing bridge. The relevant lesson is that content guidance can affect generation without replacing the entire source. Do not add an extra generator, oracle guidance, or second encoder solely to imitate it.
