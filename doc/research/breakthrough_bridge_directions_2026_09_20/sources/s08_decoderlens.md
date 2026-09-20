# Langedijk et al. 2024 — DecoderLens

Source: https://aclanthology.org/2024.findings-naacl.296.pdf
Venue: Findings of NAACL 2024, pp. 4764–4780.

## Verbatim evidence (short excerpt)

- Abstract, p. 4764: “cross-attend representations of intermediate encoder activations.”

The paper uses intermediate encoder activations through the decoder cross-attention interface without additional training, supporting the feasibility of exposing depth taps while making no performance claim for summarization.

## Mechanism relevance

Although DecoderLens is an interpretability method rather than a proposed summarization bridge, it is independent evidence that intermediate encoder activations can be consumed through the decoder cross-attention interface. It supports exposing depth taps as a controlled experimental axis; it does not support a claim of improved ROUGE by itself.
