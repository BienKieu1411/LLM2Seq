# S09 — User-supplied aggregate PubMed evaluation results

**Type:** first-party observational evidence, 2026-09-18. **Credibility 3/5, recency 5/5, bias risk 3/5.** Raw full-vs-control paired predictions and checkpoint metadata were not supplied in this environment.

**Short source quotation:** “ROUGE-1=49.657 ROUGE-2=22.098 ROUGE-L=45.920” (user report for revised contextual-value bridge).

Earlier user-reported full AFMR was `49.686/22.153/45.939`; direct projection was `49.671/22.135/45.956`. These scores show the existing full advantage is tiny and mixed, and the added contextual-value run is worse than direct projection on all three metrics. Without paired predictions and full run configuration, they do not reveal whether the cause is bridge architecture, training variance, checkpoint choice, or evaluation details.
