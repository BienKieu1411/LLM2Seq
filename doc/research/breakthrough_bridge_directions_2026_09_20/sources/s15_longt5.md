# Guo et al. 2022 — LongT5

Source: https://aclanthology.org/2022.findings-naacl.55.pdf  
Venue: Findings of NAACL 2022.

## Verbatim evidence

- Section 3.1.2 constructs “transient global tokens” from fixed input blocks.
- The global path is added while the original local token representations remain
  directly available.

## Relevance and limitation

LongT5 supplies summarization-specific evidence that global entries can coexist
with exact token paths. Its PubMed/arXiv results also depend on long-sequence
pretraining and an encoder architecture change. A side-memory AFMR bridge cannot
inherit those gains by architecture alone.
