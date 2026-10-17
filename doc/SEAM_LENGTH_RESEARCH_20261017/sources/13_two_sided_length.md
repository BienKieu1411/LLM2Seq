# A Method of Integrating Length Constraints into Encoder-Decoder Transformer for Abstractive Text Summarization

- Authors: Ngoc-Khuong Nguyen, Dac-Nhuong Le, Viet-Ha Nguyen, Anh-Cuong Le.
- Citation: Intelligent Automation & Soft Computing 38(1):1-18, 2023; publisher lists issue publication 2024-01-26.
- DOI: 10.32604/iasc.2023.037083.
- URL: https://www.techscience.com/iasc/v38n1/55289/html
- Accessed: 2026-10-17, full HTML inspected.
- Type: original journal paper.
- Credibility: usable precedent; limited confidence in comparative claims. Recency: older. Bias: author experiments; ablation prose contains an inconsistency about model 2, so do not infer details from that paragraph alone.

Verified excerpt, section 3: “length embedding is integrated with word embedding in both the encoder and decoder.”

Two-sided length conditioning predates this proposal. The method changes embedding/attention behavior, unlike prompt adaptation. It uses smaller Transformers with word embeddings on English corpora. Its discussion also acknowledges a competing method's better R2 on NEWSROOM. Use it for prior art, not a guarantee that SEAM's R2 improves.
