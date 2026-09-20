# Mohtashami & Jaggi 2023 — Landmark Attention

Primary record: https://proceedings.neurips.cc/paper_files/paper/2023/file/ab05dc8bf36a9f66edbff6992ec86f56-Paper-Conference.pdf  
Venue: NeurIPS 2023, pp. 37–55.

## Verbatim evidence

- Abstract, p. 37: a landmark represents each input block and enables retrieval “directly through the attention mechanism.”
- Introduction, p. 38: the landmark “acts as a gate for attending to its corresponding block.”
- Method, pp. 40–41, Eqs. 1–4: grouped softmax gives block-level gating while retaining token-level attention inside the selected block.
- Method, p. 40: “retrieving a block instead of a single token” preserves local context around the selected item.

## Mechanism relevant to EviSeq

Append one lightweight landmark entry per fixed source block to the bridge memory. The decoder query scores landmarks first; the landmark score gates attention to the block’s original token K/V. The bridge therefore supplies a block index while grounded copy and lexical values remain attached to the original H0 tokens.

This is distinct from evidence slots: each landmark has a fixed source span and is only a gate/index. It is also distinct from region residuals: no pooled vector is added to the original token key.

## Training status and limitation

The paper trains landmark tokens with the standard language-model training procedure and an altered attention normalization; it does not introduce a summarization auxiliary loss. It is causal language modeling, so the landmark placement and grouped-softmax behavior require a careful encoder-decoder adaptation. The original paper also warns that compressed/gist memories can lose details; retaining exact source blocks is the protection.
