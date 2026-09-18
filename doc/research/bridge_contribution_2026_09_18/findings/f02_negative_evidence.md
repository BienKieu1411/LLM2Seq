# F02 — Counterevidence and failure modes

The positive papers do not justify a universal hierarchy claim. The following counterevidence should shape the implementation and evaluation.

| Counterexample | What happened | Consequence for AFMR |
| --- | --- | --- |
| PROM on arXiv | A flat phrase-copy auxiliary improved BART by only +0.06 R-1, +0.08 R-2, and -0.04 R-L on the reported arXiv table, much smaller than its WikiHow gains (S05). | Copy supervision alone is not evidence that a long-document bridge will contribute. The bridge must reach query-conditioned source selection. |
| CODI SpanCopy + GR | On original data, adding Global Relevance reduced R-2 on CNN/DailyMail (20.86→20.61), XSum (22.76→22.36), and PubMed (19.86→19.82); its overall average ROUGE change was worse than SpanCopy alone (S06). | A source prior can over-focus copying and suppress legitimate generated content. Use a soft, calibrated prior and keep the full/original distribution in validation. |
| MiddleSum Multi-News | Llama-2-7B hierarchical inference raised PubMed R-2 10.97→13.36 but lowered Multi-News 10.43→7.06; Llama-2-13B also fell 10.38→6.71 on Multi-News (S07). | Domain-specific gains are plausible; report Multi-News or another multi-document stress slice and do not claim generality from PubMed. |
| Silver-label mismatch | Deutsch & Roth's oracle silver rows are far above end-to-end predicted-span rows. Their noise experiment shows end-to-end performance can improve when training labels are made noisier, because that better matches predicted spans (S04). | The candidate must train with its predicted-prior path and audit train/eval mismatch. Do not evaluate with gold evidence masks. |

These failures also rule out a hard top-k source mask as the default. AFMR currently preserves every visible source token and uses a soft additive prior. A proposed evidence prior should preserve that property, allow the LM branch to generate novel content, and expose uncertainty rather than forcing one source occurrence.

