# S04 — Landmark Attention

**Primary source:** Mohtashami and Jaggi, *Landmark Attention: Random-Access Infinite Context Length for Transformers*, NeurIPS 2023.

URL: https://arxiv.org/abs/2305.16300

## Evidence

- The abstract identifies a failure mode of recurrent memory and separate retrieval: they can compromise random-access attention or become incompatible with the model's attention.
- The proposed mechanism uses a landmark token for each input block and trains attention to select relevant blocks through the model's own attention. The short source wording is: “uses a landmark token to represent each block.”
- The paper reports access to complete context, random-access flexibility, and LLaMA-7B context extension beyond 32k tokens.

## Transfer to AFMR

Create a low-cost block landmark bank from `H0`. At each decoder step, first compute a soft distribution over landmarks. Use it as a differentiable routing distribution for an exact-token read from the corresponding original `K0,V0` blocks. Add that retrieved context through a bounded side gate while retaining the ordinary full-source cross-attention. Grounded copy sees no landmark score, so occurrence-level copying is preserved.

## Caveats

Landmark Attention targets context extension and decoder-only language modeling, not abstractive summarization. Hard top-k retrieval can discard evidence and create train/test differences. Use soft routing or a straight-through-free top-k approximation first, and log the mass assigned to the fallback full-token route.

