# S08 — Gated Attention for Large Language Models

URL: https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html
Authors: Qiu et al. Venue/year: NeurIPS 2025. Accessed: 2026-09-23.
Type: primary research. Credibility 5/5; recency 5/5; advocacy risk medium.
Verified excerpt, abstract: “applying a head-specific sigmoid gate after the Scaled Dot-Product Attention”.

Strong evidence for a particular gating location in pretrained-from-scratch
language models. Adding a source bridge before K/V is a different computation;
it does not inherit the same argument about nonlinearity between V and O.
Changing the decoder output gate is outside the user's requested bridge scope.
