# Source 02 — So et al. (2021/2022), Primer

- URL: https://arxiv.org/abs/2109.08668
- Full text: https://arxiv.org/html/2109.08668v2
- Authors/venue: David R. So, Wojciech Mańke, Hanxiao Liu, Zihang Dai, Noam Shazeer, Quoc V. Le; NeurIPS 2021; arXiv v2 revised January 2022.
- Proceedings: https://proceedings.neurips.cc/paper_files/paper/2021/hash/2f3c6a4cd8af177f6456e7e51a916ff3-Abstract.html
- Accessed: 2026-09-23.
- Source type: primary architecture-search paper.
- Transfer confidence: local-operator precedent; low transfer to a cross-tokenizer encoder-decoder bridge.

## Verified evidence

Primer reports two main modifications: squared ReLU and depthwise convolution after Q, K, and V projections. The paper specifies a width-3 spatial depthwise convolution per attention head and states that pointwise projection followed by depthwise convolution was more effective than the reverse ordering. Its compact description is: “adding a depthwise convolution layer after each Q, K, and V projection in self-attention.” (14 quoted words.) The paper also reports that wider depthwise convolutions and standard convolutions did not improve results and sometimes hurt them.

## Implication for XOV

The evidence supports a small depthwise local operator over a contiguous token stream. It does not establish that width 3 is optimal for decoder subwords, or that a convolution over decoder-token embeddings remains useful after reverse alignment into encoder positions. Primer's convolution is inside self-attention and changes Q/K/V before attention; XOV's branch is a source construction path whose values are later normalized and consumed by cross-attention. The objectives, tokenizer, masking, and training regimes differ.

## Constraint

Keep the first pilot at width 3 and depthwise, with no extra wider or full convolution. Treat any change in width or channel mixing as a separate capacity ablation; a Primer citation cannot supply a quality prediction for XOV.
