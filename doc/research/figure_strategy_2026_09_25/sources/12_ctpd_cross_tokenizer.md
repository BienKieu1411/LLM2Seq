# Nguyen et al. (2026): shared character spans

- Primary record: https://arxiv.org/abs/2601.11865
- Type: original arXiv preprint on cross-tokenizer preference distillation; accessed 2026-09-25.
- Credibility 4/5; recency 5/5; low-bias 3/5 (authors evaluate their own proposed method).
- Short quote from abstract: “maps teacher and student tokens to shared character-level spans”.
- Relevant observation: character-level spans are a legitimate common coordinate system for heterogeneous tokenizers. For SEAM the relevant test would be generation of source-visible terms under true segmentation disagreement.
- Limit: distillation is a different task from copying, and this paper does not establish that the main SEAM backbone pair has substantial mismatch or that copy improves its summaries.
