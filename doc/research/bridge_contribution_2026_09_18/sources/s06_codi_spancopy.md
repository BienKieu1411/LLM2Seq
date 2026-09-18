# S06 — CODI SpanCopy + Global Relevance counterexample

**Citation.** Wen Xiao and Giuseppe Carenini (2023), *Entity-based SpanCopy for Abstractive Summarization to Improve the Factual Consistency*, CODI 2023.

**Primary record.** [ACL Anthology landing page](https://aclanthology.org/2023.codi-1.9/) · [author PDF](https://aclanthology.org/2023.codi-1.9.pdf).

**Role.** Negative/ablation evidence for a source prior that improves a filtered setting but can over-focus copying on the original distribution.

**Short source quotation (10 words).** “it fails to deliver any gain on the original datasets.”

## What was tested

SpanCopy adds an entity-level copy path to PEGASUS. Global Relevance (GR) predicts a prior over source entities and multiplies the copy logits; a GR auxiliary loss trains the prior. The paper reports filtered datasets, where every reference entity appears in the source, and original datasets, where that condition is not enforced.

On original datasets, the GR variant's ROUGE-2 is 20.61 on CNN/DailyMail versus 20.86 for SpanCopy alone, 22.36 on XSum versus 22.76, 19.82 on PubMed versus 19.86, and 16.87 on arXiv versus 16.76. Thus GR loses R-2 on three of four original datasets. On arXiv, GR also lowers entity-summary F1 from 20.39 to 20.15 and source precision from 56.88 to 54.21. The paper's relative-average table gives GR an overall average ROUGE change of -0.15 versus -0.06 for SpanCopy alone, while factual consistency can move independently of saliency.

The authors attribute the failure mode to an overly strong focus on source entities, which can penalize new summary-worthy entities. The filtered/original split is especially relevant for silver supervision: an oracle-compatible training distribution can overstate the inference benefit.

## Design implication for `eviseq_new`

**Supported:** a predicted source prior should be regularized and evaluated on the original data distribution. Sharing a prior with copy and semantic cross-attention is a plausible architectural contribution only if it improves selection without suppressing legitimate generated content.

**Still unproven:** SpanCopy is entity-specific and does not use AFMR regions or fixed `H0` values. The R-2 losses are a warning against hard masks, overly sharp priors, or a claim that any auxiliary prior makes the full model better.

