# F2 — Compressed features should route to exact tokens, not replace them

Hierarchical and landmark attention provide a route in which a compact block
representation only selects a block and the final context still reads original
token values. This directly avoids the information-loss and write-back pattern
shared by the recent failures. Transfer to CE-only PubMed summarization remains
a falsifiable hypothesis.

Sources: `s06_selective_attention.md`, `s09_landmark_attention.md`,
`s10_hierarchical_transformer.md`, `s13_dyle_dynamic_snippets.md`,
`s20_local_evidence.md`.
