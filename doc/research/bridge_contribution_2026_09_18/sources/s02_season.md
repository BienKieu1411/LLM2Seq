# S02 — SEASON salience allocation and cross-attention

**Citation.** Fei Wang, Kaiqiang Song, Hongming Zhang, Lifeng Jin, Sangwoo Cho, Wenlin Yao, Xiaoyang Wang, Muhao Chen, and Dong Yu (2022), *Salience Allocation as Guidance for Abstractive Summarization*, EMNLP 2022.

**Primary record.** [ACL Anthology landing page](https://aclanthology.org/2022.emnlp-main.409/) · [author PDF](https://aclanthology.org/2022.emnlp-main.409.pdf). The arXiv preprint is [arXiv:2210.12330](https://arxiv.org/abs/2210.12330).

**Role.** Support for a predicted, gold-derived source salience signal that changes cross-attention while preserving the value representation.

**Short source quotation (17 words).** “the salience-aware cross-attention module is flexible to decide how much signal to accept from the salience guidance.”

## What was tested

SEASON assigns sentence salience degrees using the reference summary, trains a salience classifier, and jointly trains the summarizer. At cross-attention, a salience embedding is added to encoder hidden states used as keys; the original encoder hidden states remain the values. Section 3.3/3.4 states that generation uses gold salience embeddings at training and predicted expected embeddings at test. AFMR's proposed auxiliary differs by using a **predicted** source bias for generation in both phases; gold-derived labels only supervise the auxiliary loss. This separation is the relevant architectural analogy, not an identical training protocol.

On CNN/DailyMail, the paper's module ablation reports:

| SACA | MTL salience loss | R-1 | R-2 | R-L |
| --- | --- | ---: | ---: | ---: |
| absent | absent | 44.21 | 21.23 | 41.17 |
| absent | present | 44.57 | 21.55 | 41.49 |
| predicted | present | 46.27 | 22.64 | 43.08 |
| gold upper bound | present | 54.85 | 31.36 | 52.14 |

The reported predicted-SACA gain over the no-SACA/no-MTL row is +2.06/+1.41/+1.91 ROUGE-1/2/L. The paper also reports +0.33/+0.32/+0.60 on Newsroom and evaluates human informativeness, faithfulness, fluency, and ranking on 100 CNN/DailyMail test examples. The gold row is an upper bound, not an inference setting.

## Design implication for `eviseq_new`

**Supported as a hypothesis:** train a source salience/evidence prior from training-only reference alignment, predict it from the source-side representation, and let it influence the cross-attention keys or logits while retaining the original value anchor. This gives a bridge output a direct route to content selection.

**Still unproven for AFMR:** SEASON uses news datasets, sentence labels, a separately described salience classifier, and BART. It does not use overlapping AFMR regions, a grounded-copy head, or the Perl155/PubMed setup. The large gold upper bound warns that train/eval mismatch and noisy predicted labels can dominate the observed gain.
