# Maruf, Martins & Haffari 2019 — Selective Attention for Context-aware NMT

Primary record: https://aclanthology.org/N19-1313.pdf  
Venue: NAACL-HLT 2019, pp. 3092–3102.

## Verbatim evidence

- Abstract, p. 3092: “sparse attention to selectively focus on relevant sentences” and then attend to “key words in those sentences.”
- Introduction, p. 3093: “Hierarchical Attention ... is computed dynamically for each query word.”
- Method, pp. 3094–3095, Eqs. 3–5: sentence-level sparsemax is multiplied by within-sentence word-level sparsemax, and the resulting distribution reads word values.
- Method, p. 3095: “the output of the context layer” is integrated through a learned context gate.
- Footnote, p. 3094: residual connections after the context sublayers had “a deteriorating effect on the translation scores.”

## Mechanism relevant to EviSeq

Use deterministic source blocks or sentences as a coarse index. For each decoder query, compute a sparse block distribution, then compute a second token distribution inside each selected block. The final source attention is

`a_t(i) = a_t(block(i)) * a_t(i | block(i))`.

The value read is the original token-level source value, so the method selects exact source positions instead of adding a pooled residual to every key.

## Training status and limitation

The paper is trained as a supervised sequence-to-sequence likelihood model; no oracle span loss is required for the hierarchical attention itself. It is NMT rather than summarization, so a ROUGE gain is only a transfer hypothesis. Sparsemax can become too selective; retain a full-attention fallback and initialize the new route at zero.
