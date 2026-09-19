# S1 — TopDownFormer

Primary source: https://aclanthology.org/2023.findings-eacl.94.pdf, Table 1 and Section 3.7.

It combines coarse segment representations with token representations. PubMed AvgPool is 48.34/21.40/44.22 and AdaPool is 51.05/23.26/46.47 (R-1/2/L). This is evidence that *pooling design matters in that model*, not that AFMR's 128-token means caused its score decrease. AdaPool also trains an importance tagger using labels derived from references; it is not a pure, directly comparable architecture change. Their checkpoint selection uses validation R-2, whereas the local candidate scripts use last.pt.
