# g01 — Transformer attention exposes the equivalence boundary

- **Primary source:** Vaswani et al., *Attention Is All You Need* (NeurIPS 2017).
- **URL:** https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
- **Web fetch:** 2026-09-23; PDF lines 112–153 and 182–188.
- **Role:** Formal baseline for what decoder cross-attention already does.
- **Credibility:** 5/5 primary architecture paper. **Recency:** foundational, not current. **Transfer:** exact attention algebra transfers; task and backbones do not.

**Verified short quote (8 words):** “linearly project the queries, keys and values”

The paper defines scaled dot-product attention as softmax of query–key scores
times values. In encoder–decoder attention, decoder queries read all encoder
positions. The key/value projections are linear and position-wise: for a memory
row (M_i), (K_i=W_KM_i) and (V_i=W_VM_i) do not depend on (M_{i-1}) or
(M_{i+1}). The softmax can mix rows after their keys and values are formed,
but it does not make row (i)'s key a function of neighboring source rows.

That gives the clean algebraic test for a bridge contribution. A fixed linear
map (P(H_i)) or a per-token MLP can be compared with the existing projections;
a position-mixing map (M_i=f(H_{i-r:i+r})) has a nonzero cross-position
derivative and cannot be folded into a per-position (W_K/W_V) for every input.
This is an expressivity distinction, not a claim that the full decoder cannot
learn an equivalent function after several attention layers.

The source does not study frozen PPLX encoder states, Qwen decoder adaptation,
grounded copying, or summarization. It supports only the interface algebra and
the no-duplicate criterion.
