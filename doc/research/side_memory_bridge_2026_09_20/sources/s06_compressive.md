# S06 — Compressive Transformer

**Primary source:** Rae et al., *Compressive Transformers for Long-Range Sequence Modelling*, ICLR 2020.

URL: https://arxiv.org/abs/1911.05507

## Evidence

- Section 3 describes fine-grained memory plus a second compressed memory. The abstract formulation is that the model “maps past hidden activations to a smaller set of compressed representations.”
- The model attends over both granular and compressed memories; the compressed bank is stored instead of discarding old activations.
- On PG-19, the paper reports 33.6 perplexity versus 36.3 for Transformer-XL at the matched setup, and its analysis reports a larger benefit for rare words than frequent words.
- The conclusion explicitly warns that compression is unlikely to help when a task lacks long-range reasoning, which is a relevant negative control for PubMed.

## Transfer to AFMR

Build two side banks from source chunks: a fine evidence bank for local facts and a coarser global bank for document-level relations. Let the decoder read both through a gated side branch, while `K0,V0` and grounded copy remain untouched. This gives a principled dual-timescale memory without altering token-key matching.

## Caveats

The paper uses auxiliary reconstruction/attention losses in some settings and recurrent unrolling. Those are deliberately excluded from the first AFMR experiment to preserve the CE-only claim. A pure CE version may undertrain the compression function; if so, that is evidence against the direction under the user's constraints.

