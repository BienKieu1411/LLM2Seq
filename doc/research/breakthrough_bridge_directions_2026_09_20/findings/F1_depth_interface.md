# F1 — Encoder depth should be selected at the decoder interface

Local AFMR mixes depth once for all decoder layers. Four independent sources
support exposing intermediate encoder representations at decoder cross-attention,
with layer-specific selection. The transferable conclusion is architectural
feasibility. Evidence that this improves PubMed ROUGE is still insufficient.

Sources: `s01_composition.md`, `s02_layerwise_coordination.md`,
`s03_dense_information_flow.md`, `s08_decoderlens.md`, `s20_local_evidence.md`.
