# Zhang et al. 2018 — Document-Level Context Encoder

Source: https://aclanthology.org/D18-1049/
Venue: EMNLP 2018, pp. 533–542.

## Verbatim evidence (short excerpt)

- Abstract: “a new context encoder to represent document-level context.”

The abstract states that this context representation is incorporated into both the original encoder and decoder, with gains reported on two document-level translation settings.

## Mechanism

Use a document representation as a separate conditioning stream for both the source bridge and decoder. For the current model, the safer adaptation is to expose the document controller only to a gated decoder-side cross-attention branch while preserving the token-aligned PPLX memory.

## Relevance and limitation

This is a document-context precedent rather than a PPLX/Qwen connector. The original work uses document-level parallel translation context and a two-step training setup, so its reported gains do not establish a PubMed summarization gain.
