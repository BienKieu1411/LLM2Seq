# Region-query failure audit: research plan

Question: why did both decoder-query region bridges underperform the PubMed controls, and which single architectural hypothesis is worth testing next?

Evidence tiers: (1) user-provided, same-name aggregate ROUGE-1.5.5 results; (2) repository code, configs and tiny tests; (3) primary papers on hierarchical source selection. No checkpoint, predictions, validation curves or paired outputs for these runs are present locally. Therefore causal attribution and any promised ROUGE improvement are out of scope.

Decision rule: preserve the three existing architectures; implement one isolated candidate that keeps token-level values and grounded copy, uses only CE, and offers a post-hoc branch-off diagnostic. Compare matched checkpoints/configs/seed/data and paired predictions on validation before drawing a paper claim. Use test only after freezing the candidate.
