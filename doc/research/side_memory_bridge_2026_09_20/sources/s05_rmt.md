# S05 — Recurrent Memory Transformer

**Primary source:** Bulatov, Kuratov, and Burtsev, *Recurrent Memory Transformer*, 2022.

URL: https://arxiv.org/abs/2207.06881

## Evidence

- The abstract says the method adds special memory tokens to the input or output sequence without changing the Transformer model.
- Figure 1 description states that memory output is passed to the next segment and gradients can flow from the current segment through memory to the previous segment.
- The authors report parity with Transformer-XL at smaller memory sizes and improvements on tasks requiring longer sequence processing.

## Transfer to AFMR

Partition the source into fixed chunks, run a small bridge memory updater over chunk summaries, and expose the resulting memory bank as an additional decoder cross-attention source. Preserve the complete source-token bank for the current document and keep copy on it. For a first CE-only implementation, use one forward pass over the chunks with bounded recurrent updates; do not add BPTT reconstruction or generated candidate text.

## Caveats

RMT's main setting is segment recurrence for autoregressive language modeling. A recurrent bank can lose exact biomedical details and is more difficult to batch than a one-shot resampler. It is a lower-priority experiment unless validation shows a long-document weakness that a static bank cannot address.

