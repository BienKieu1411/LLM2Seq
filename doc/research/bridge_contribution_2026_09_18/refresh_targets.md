# Refresh targets

Research snapshot: 2026-09-18. Refresh the literature package before using it in a paper or after any AFMR training result changes the proposed design.

| ID | Refresh URL | Trigger |
| --- | --- | --- |
| S01 | https://aclanthology.org/2023.findings-eacl.94/ | Check if a revised paper or official code changes the top-down ablation. |
| S02 | https://aclanthology.org/2022.emnlp-main.409/ | Recheck salience-aware key/value details and the SACA ablation. |
| S03 | https://aclanthology.org/2022.emnlp-main.355/ | Recheck graph-selection and no-hierarchy ablations. |
| S04 | https://aclanthology.org/2023.eacl-main.42/ | Recheck silver-span noise and train/inference mismatch findings. |
| S05 | https://aclanthology.org/2024.lrec-main.1148/ | Recheck the arXiv flat-copy comparison and any final-version corrections. |
| S06 | https://aclanthology.org/2023.codi-1.9/ | Recheck original versus filtered GR results, especially PubMed and arXiv. |
| S07 | https://aclanthology.org/2024.acl-long.153/ | Recheck MiddleSum Multi-News results and whether code/data revisions affect the comparison. |
| S10 | https://aclanthology.org/2022.acl-long.118/ | Keep the arXiv cross-system comparison distinct from GovReport/QMSum objective ablations. |
| S11 | https://aclanthology.org/2023.eacl-main.121/ | Keep full-source bigram overlap separate from AFMR visible-source coverage. |
| S12 | https://aclanthology.org/2024.findings-acl.389/ | Recheck proposition-level alignment scope and Multi-News limitation. |
| S13 | https://aclanthology.org/2025.findings-emnlp.846/ | Recheck the cross-encoder attribution findings before claiming lexical labels miss a given AFMR example. |
| S14 | https://aclanthology.org/2024.fever-1.24/ | Keep medical evidence-extraction metrics distinct from PubMed summary ROUGE. |
| S15 | https://aclanthology.org/2022.tacl-1.36/ | Recheck the two-view decoder mechanism before making a novelty claim for a query-focused route. |

The next refresh should add AFMR-specific evidence only after the three-way paired runs: checkpoint logs, prior/gradient diagnostics, validation and held-out scores, and seed intervals. Do not replace the current external-vs-unknown distinction with a score from a single run.
