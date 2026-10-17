# F1: requested length belongs in the training input

Status: supported as a design precedent; improvement in SEAM untested.

Independent primary precedents: [Kikuchi](../sources/01_kikuchi.md), [Liu](../sources/02_liu.md), [Takase](../sources/03_takase.md). [Author code](../sources/06_larft_code.md) corroborates a modern input contract but is not an independent validation of LARFT.

For SEAM, match W to the actual supervised summary, render W in the decoder instruction, and preserve it through grouping and scoring. Word counts and decoder token caps are separate units. The literature does not establish a best Vietnamese template or guarantee compliance.

Falsifier: matched instructions fail to improve requested-length error on held-out validation relative to the same checkpoint and inference budget.
