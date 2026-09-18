# S11 — LONGEVAL and the limit of exact lexical evidence

**Citation.** Krishna et al. (2023), *LONGEVAL: Guidelines for Human Evaluation of Faithfulness in Long-form Summarization*, EACL 2023. [Primary paper](https://aclanthology.org/2023.eacl-main.121.pdf).

**Role.** Empirical warning against assuming exact source-summary n-gram overlap captures all salient PubMed evidence.

**Short source quotation (12 words).** “54% of summary bigrams are present in the source” (PDF p. 4).

## What was tested

Section 3 (PDF p. 4) reports that 54% of bigrams in human PubMed summaries appear in the full source, whereas outputs of LongT5 and BigBird-PEGASUS have 87% and 74% overlap. LONGEVAL studies faithfulness evaluation, not a supervised salience loss. The figures concern its PubMed sample and full source, not the first 4096 tokens visible to AFMR. They do not mean the remaining 46% is unsupported; paraphrase, morphology, and wording changes are possible.

## Implication for AFMR

The current bigram/trigram exact-match label can miss valid paraphrased evidence. Audit label precision and coverage on the actual *visible source* before selecting a loss coefficient. Do not claim these paper percentages measure the AFMR training labels.
