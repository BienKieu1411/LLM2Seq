# F3 — A compact memory needs an independent normalization path

Multi-source attention and gated side-attention papers support computing a
separate context and fusing outputs after attention. This prevents compact
entries from stealing token attention mass and prevents a compressed vector from
rewriting lexical keys. Cross-domain transfer is substantial, so the side path
is ranked behind exact-depth and exact-token candidates.

Sources: `s04_fusion_in_decoder.md`, `s14_flamingo.md`, `s15_longt5.md`,
`s16_gtca.md`, `s20_local_evidence.md`.
