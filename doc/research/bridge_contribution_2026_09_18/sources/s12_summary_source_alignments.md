# S12 — Proposition-level summary-source alignment

**Citation.** Ernst et al. (2024), *The Power of Summary-Source Alignments*, Findings of ACL 2024. [Primary paper](https://aclanthology.org/2024.findings-acl.389.pdf).

**Role.** Evidence that source-summary correspondence can require units finer than sentences and richer than lexical matching.

**Short source quotation (12 words).** “applying it at the more fine-grained proposition span level” (abstract, PDF p. 1).

## What was tested

The authors annotate source-summary alignments at proposition level over Multi-News, a multi-document dataset, then derive six tasks including salience detection and evidence detection (PDF pp. 1–5). Their annotation guidelines explicitly account for paraphrases; one warning is that sentence-level heuristics can misalign information. They do not test AFMR, a PubMed training loss, or grounded copy.

## Implication for AFMR

It supports treating exact overlap as a weak *proxy*, with manual checks for occurrence ambiguity and paraphrase. It does not justify claiming that the AFMR lexical labels are proposition-aligned or that an alignment loss improves ROUGE.
