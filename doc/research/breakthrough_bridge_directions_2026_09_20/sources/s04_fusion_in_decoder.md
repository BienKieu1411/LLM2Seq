# Izacard & Grave 2021 — Fusion-in-Decoder

Source: https://aclanthology.org/2021.eacl-main.74.pdf
Venue: EACL 2021, pp. 874–880.

## Verbatim evidence (short excerpt)

- Sec. 3, p. 876: “The model thus performs evidence fusion in the decoder only.”

The method independently encodes passages and concatenates their representations for decoder cross-attention. The authors report that increasing the number of passages improved their QA scores, while the architecture scales linearly in the number of separately encoded contexts.

## Mechanism

Create a separate document-level or chunk-level memory bank, but concatenate it with the untouched token memory only at the decoder cross-attention interface. Keep source-token positions and grounded-copy values in the original bank; document/chunk memories are additional K/V entries with a segment/type bias.

## Relevance and limitation

This avoids the failed region/slot write-back pattern and preserves exact token access. PubMed examples already have one source document, so the bridge can chunk the PPLX sequence and add independently pooled chunk memories. FiD itself is retrieval QA, not summarization; using it as a source-document bridge is an architectural transfer that must be validated.
