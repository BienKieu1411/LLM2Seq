# S15 — Query-focused and original document views

**Citation.** Xu and Lapata (2022), *Document Summarization with Latent Queries*, TACL 10. [Primary paper](https://aclanthology.org/2022.tacl-1.36.pdf).

**Role.** Relevant prior art for decoder use of a query-focused view and a query-agnostic original document view; an explicit novelty counterweight.

**Short quotation (13 words).** “the decoder will first attend to signals coming from query Q” (PDF p. 6).

## Claim and evidence

The paper represents queries as latent variables over document tokens and jointly trains a query model with a conditional language model. Section 5 (PDF pp. 6–7) describes a decoder that sequentially attends to a query-focused document view and the original document view. It uses BART-based encoders and a different training objective, not an AFMR region read or grounded-copy mixture.

## AFMR implication

The broad idea of combining a focused view with original document context is **not novel by itself**. A proposed AFMR paper would need to specify its narrower distinction—decoder-step-conditioned soft regional read over full-source keys, `H0`-anchored values, and grounded copy—and test each route. The existence of this paper does not imply the AFMR variant will improve PubMed ROUGE.
