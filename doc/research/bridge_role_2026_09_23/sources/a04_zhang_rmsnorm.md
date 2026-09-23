# Source A04 — Root Mean Square Layer Normalization

- Authors: Biao Zhang, Rico Sennrich
- Version: arXiv:1910.07467, 2019
- URL: https://arxiv.org/abs/1910.07467
- Type: primary normalization study
- Quality: high for RMSNorm’s stated invariances and comparisons; not a bridge-specific study

## Verified evidence

RMSNorm removes LayerNorm’s mean-centering step and retains RMS-based
rescaling. The paper reports comparable performance to LayerNorm with lower
runtime in its evaluated architectures and describes the resulting rescaling
invariance. This matters because AFMR’s decoder cross-attention applies an
existing `memory_norm` before K/V projections, and Qwen attention also has a
key normalization copied from self-attention.

Short verbatim excerpts (under the source quote limit):

> “re-centering invariance in LayerNorm is dispensable”

> “RMSNorm ... giving the model re-scaling invariance”

## Use in this audit

Do not add another RMSNorm or LayerNorm to the bridge in the first probe. A
uniform scale change may be attenuated by the existing normalization, while a
neighbor-difference direction changes feature orientation and can survive. The
`Convolution for Large Language Models` report independently found that added
normalization after the relevant Qwen3 convolution was worse than a residual
shortcut. That is still not proof for AFMR, so the probe must measure
`||Norm(M_local)-Norm(H0)||`, key-logit change, and CE sensitivity.
