# Libovický et al. 2018 — Multi-source Transformer decoder

Source: https://aclanthology.org/W18-6326.pdf  
Venue: WMT 2018.

## Verbatim evidence

- The paper evaluates “serial, parallel, flat, and hierarchical” source
  combination strategies.
- In the parallel form, the decoder computes attention to each source
  independently before combining context vectors.

## Relevance and limitation

The independent-softmax construction supports a side-memory branch that cannot
steal probability mass from the exact token bank. The paper studies translation
with genuinely different sources, not two representations of one PubMed source,
so it establishes mechanism feasibility rather than expected ROUGE gain.
