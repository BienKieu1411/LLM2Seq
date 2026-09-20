# Mahabadi et al. 2021 — HyperFormer

Source: https://aclanthology.org/2021.acl-long.47.pdf
Venue: ACL-IJCNLP 2021, pp. 565–576.

## Verbatim evidence (short excerpt)

- Abstract, p. 565: “condition on task, adapter position, and layer id.”

The paper uses a shared hypernetwork to generate layer-specific adapter parameters from a conditioning embedding. Sec. 2 formulates the training loss with the usual cross-entropy likelihood; Secs. 2.3–2.4 add layer and adapter-position embeddings.

## Mechanism

Replace the task embedding with a compact document/controller embedding derived from PPLX. A shared hypernetwork emits a small per-decoder-layer low-rank modulation for the cross-attention projections (or a per-layer prefix K/V generator). Apply it to a complete K/V pair, with the direct projection path retained as an explicit base and a bounded gate initialized near zero.

## Relevance and limitation

The paper proves the conditioning/hypernetwork pattern and CE-compatible training, not document-conditioned summarization. Generating full projection matrices would be too expensive; only rank-16/32 factors or per-head scale/shift should be generated. The document vector must be detached from grounded-copy preparation so copy alignment stays unchanged.
