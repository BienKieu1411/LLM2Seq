# S05 — PROM as a flat long-document copying counterexample

**Citation.** Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan (2024), *PROM: A Phrase-level Copying Mechanism with Pre-training for Abstractive Summarization*, LREC-COLING 2024. The earlier public version is [arXiv:2305.06647](https://arxiv.org/abs/2305.06647).

**Primary record.** [ACL Anthology landing page](https://aclanthology.org/2024.lrec-main.1148/) · [author PDF](https://aclanthology.org/2024.lrec-main.1148.pdf).

**Role.** Negative control: a flat phrase-copying auxiliary can help, but its reported long-document gain is very small.

**Short source quotation (11 words).** “Gains on CNN/DM and WikiHow are larger than on longer sequences.”

## What was tested

PROM adds an indicator layer for source n-grams and an auxiliary copying loss to a Transformer summarizer. It does not introduce a hierarchical region representation. The paper evaluates CNN/DailyMail, WikiHow, and arXiv; arXiv is described as a 113k-paper long-document benchmark.

In the WikiHow/arXiv scalability table, the flat PROM versus their BART-large results are:

| Dataset | BART R-1/R-2/R-L | PROM R-1/R-2/R-L | PROM minus BART |
| --- | --- | --- | --- |
| WikiHow | 45.22/20.13/43.73 | 45.57/20.53/44.09 | +0.35/+0.40/+0.36 |
| arXiv | 45.18/16.87/39.42 | 45.24/16.95/39.38 | +0.06/+0.08/-0.04 |

The authors also report larger copying/factuality effects on CNN/DailyMail and say the gains are smaller on longer sequences. This makes PROM useful as a flat-copy comparator, not as evidence that a bridge is unnecessary or sufficient.

## Design implication for `eviseq_new`

**Supported:** a copy-focused auxiliary objective is not a reliable proxy for a long-document bridge contribution. A full bridge should demonstrate a route beyond phrase copying, for example query-conditioned region selection and evidence use by both semantic read and copy.

**Still unproven:** PROM uses a different backbone, data, pretraining, and copy parameterization. The arXiv gap is not an AFMR ablation and cannot establish that AFMR's bridge will help or hurt.

