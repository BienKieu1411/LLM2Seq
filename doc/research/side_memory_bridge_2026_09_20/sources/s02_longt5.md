# S02 — LongT5

**Primary source:** Guo et al., *LongT5: Efficient Text-To-Text Transformer for Long Sequences*, Findings of NAACL 2022.

URL: https://aclanthology.org/2022.findings-naacl.55.pdf

## Evidence

- Page 1, Section 1, describes ETC's secondary global memory as a route through which long-input tokens interact indirectly. The paper summarizes the role as “adds a secondary input called the global memory.”
- Pages 1–2, Section 3.1.2, construct one transient global token per fixed block by aggregating the block and let each token attend locally plus to all global tokens. The global tokens are made on the fly and discarded after the attention operation.
- The paper reports state-of-the-art results on arXiv and PubMed summarization in its contribution list and plots, but the encoder architecture and PEGASUS-style pretraining are part of that result.
- The global-token section makes the key safety property explicit: the original input tokens remain present and can attend to local neighborhoods; the global path is additional context.

## Transfer to AFMR

Construct deterministic or lightly learned block evidence tokens `G_i` from `H0`, preferably by masked normalized pooling with block positions. Add a gated `Q -> G` side branch while retaining full `Q -> (K0,V0)` attention and grounded copy. A stronger variant uses `G` only to produce a soft block distribution, then reads the exact original token K/V in those blocks; the full native route remains as a fallback.

## Caveats

LongT5 changes encoder self-attention and uses long-sequence pretraining. A block mean can still blur names, numbers, negation, and occurrence identity. The AFMR transfer should therefore never replace token K/V, and its first test should use a small gate and CE only.

