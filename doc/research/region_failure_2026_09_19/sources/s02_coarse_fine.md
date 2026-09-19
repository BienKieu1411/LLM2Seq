# S2 — Ling and Rush, coarse-to-fine attention

Primary source: https://aclanthology.org/W17-4505.pdf, abstract and Sections 4–6.

The paper separates chunk attention from word attention and shows how a coarse decision can guide a fine read. Its *hard* chunk-selection models lagged standard attention despite yielding sparse reading. It supports studying a soft region prior while retaining all source tokens; it does not establish that such a prior will improve AFMR or that hard selection is safe.
