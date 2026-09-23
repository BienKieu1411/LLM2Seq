# Source 04 — Touvron et al. (2021), LayerScale

- URL: https://arxiv.org/abs/2103.17239
- Full text: https://arxiv.org/html/2103.17239
- Authors/venue: Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, Hervé Jégou; ICCV 2021.
- Accessed: 2026-09-23.
- Source type: primary optimization/initialization paper.
- Transfer confidence: useful alternative initialization evidence; task and modality differ.

## Verified evidence

LayerScale multiplies each residual-block output by a learnable diagonal matrix initialized to a small positive value. The authors define the change as weights “initialized close to (but not at) 0.” (7 quoted words.) Their ablation reports that a small nonzero initialization can converge better than exact zero in their deep vision-transformer setting; the paper also keeps normalization and warmup when adapting related methods.

## Implication for XOV

This is a counterpoint to treating zero initialization as universally best. A small per-channel residual scale with a nonzero branch output can preserve a near-identity function while allowing internal gradients immediately. Adding such a scale while leaving lexical_up exactly zero would not remove XOV’s initial gradient delay. The transfer is limited: LayerScale was tested in deep image transformers, not a single lexical bridge, and its favorable initialization does not prove better summarization.

## Constraint

If the primary XOV pilot shows a cold-start or nearly inert branch, test one small-epsilon residual-scale variant with the same architecture and optimizer. Keep the cap and copy routing fixed; record direct-control parity separately because a nonzero branch perturbs the initial output.
