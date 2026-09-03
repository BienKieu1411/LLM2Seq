# EviSeq v2 versus T5Gemma on PubMed

## Correct benchmark

The relevant PubMed numbers are:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| T5Gemma | 49.580 | 21.990 | 45.463 |
| EviSeq PCEB | 49.228 | 21.644 | 45.312 |

EviSeq PCEB is behind by `0.352`, `0.346` and `0.151` points respectively.
That is close, but it is not a win.  The implementation therefore treats
source use and factual target conditioning as the optimization target instead
of claiming that a small metric gap is already statistically meaningful.

## Architectural diagnosis

T5Gemma 2 adapts Gemma 3 to an encoder-decoder model with UL2-style training
and provides matched 1B encoder and 1B decoder components.  EviSeq PCEB uses
a 0.6B PPLX source encoder, a 0.6B causal Qwen decoder and newly copied
cross-attention modules.  The EviSeq graph is smaller and preserves useful
pretrained components, but the decoder has not received the same large-scale
seq2seq interface adaptation.

The source encoder is also an embedding model rather than a summarization
encoder-decoder.  This makes its source representation useful, but does not
guarantee that its token memory is already aligned with target generation.
These conclusions follow from the [T5Gemma model card](https://huggingface.co/google/t5gemma-2-1b-1b),
the [T5Gemma paper](https://arxiv.org/abs/2512.14856), and the
[Qwen3 Embedding report](https://arxiv.org/abs/2506.05176).

## Changes made for the next PubMed run

The fixed internal PubMed experiment adds:

- target-free source/prompt InfoNCE;
- hard in-batch counterfactual source-swap ranking under the same target;
- identity-initialized bridge correction;
- stronger initial cross-attention gate;
- evidence loss aligned with the deployed attention prior;
- mild label smoothing and lower, separate encoder/decoder learning rates.

The source-swap branch is training-only and deliberately does not change the
one-encoder, one-bridge, one-decoder inference graph.  Its diagnostics expose
whether the correct source gets lower target NLL than the hardest wrong source.

## Reproducibility

The run is defined by
`configs/models/pplx_pubmed_pceb_corrected.yaml` and its inherited YAML
recipes. Training writes `resolved_config.yaml` beside each checkpoint;
evaluation refuses a checkpoint created by a different model/data/decoding
protocol. The optional source-swap recipe can be represented by extending the
same configuration contract without changing the deployed graph.

All future comparisons must use the same PubMed test rows, normalization,
decoding constraints and ROUGE implementation for both models.
