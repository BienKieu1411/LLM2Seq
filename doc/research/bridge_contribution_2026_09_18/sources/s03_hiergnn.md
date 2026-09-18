# S03 — HierGNN latent hierarchy and graph selection

**Citation.** Yifu Qiu and Shay B. Cohen (2022), *Abstractive Summarization Guided by Latent Hierarchical Document Structure*, EMNLP 2022.

**Primary record.** [ACL Anthology landing page](https://aclanthology.org/2022.emnlp-main.355/) · [author PDF](https://aclanthology.org/2022.emnlp-main.355.pdf).

**Role.** Support for jointly learning document structure and using a source-selection signal during decoding.

**Short source quotation (11 words).** “a graph-selection attention mechanism serves as a source sentence selector.”

## What was tested

HierGNN learns a latent sentence hierarchy with sparse matrix-tree computation, propagates sentence information over the graph, fuses the result, and uses graph-selection attention while decoding. On XSum, the reported HierGNN-PGN LIR full model has ROUGE-1/2/L and BERTScore of 30.24/10.43/24.20/27.36. Removing the HierGNN module changes these by -0.54/-1.22/-0.96/-4.20; removing graph-selection attention changes them by -0.41/-0.41/-0.17/-0.27; removing graph fusion changes them by -0.94/-0.81/-0.77/-1.39. Their CNN/DailyMail and XSum experiments also compare pretrained BART variants and human relevance, informativeness, and redundancy ratings.

The paper's analysis associates graph-selection attention with greater source coverage and shorter copied sequences on CNN/DailyMail (15.22% coverage and 16.80 copy length with GSA versus 13.74% and 18.88 without GSA). These are model-specific diagnostics, not AFMR measurements.

## Design implication for `eviseq_new`

**Supported as a hypothesis:** a learned structural prior is most defensible when it reaches the decoder selection operation, rather than existing only as an auxiliary encoder feature. A bridge prior shared by semantic cross-attention and grounded copy follows this causal pattern more closely than a disconnected prediction head.

**Still unproven for AFMR:** HierGNN operates over sentence nodes and a pointer-generator/BART stack. Its no-module ablation changes many graph components at once, and it does not test AFMR's fixed `H0` values, overlapping regions, decoder-query dependence, or copy-token marginalization. The reported margins therefore cannot be transferred numerically.

