# S04 — QA-derived silver spans for query-conditioned evidence

**Citation.** Daniel Deutsch and Dan Roth (2023), *Incorporating Question Answering-Based Signals into Abstractive Summarization via Salient Span Selection*, EACL 2023.

**Primary record.** [ACL Anthology landing page](https://aclanthology.org/2023.eacl-main.42/) · [author PDF](https://aclanthology.org/2023.eacl-main.42.pdf).

**Role.** Direct evidence that reference-conditioned, phrase-level silver labels can distinguish which occurrence of the same source phrase supports the summary.

**Short source quotation (13 words).** “The QA-based approach can reason about which occurrence of an NP is salient.”

## What was tested

For each source noun phrase, the authors generate a wh-question and test whether the gold summary answers it. The resulting silver span is used to train a span classifier; predicted marked spans then condition BART generation. This is a query/target-conditioned selection signal, not a factuality annotation by humans.

On CNN/DailyMail, the end-to-end QA-span model reports ROUGE-1/2/L 45.5/21.9/42.4, BERTScore 88.5, and QAEval 24.4, versus their BART baseline 44.1/21.0/40.9, 88.3, and 23.5. On NYTimes, QA spans report 55.2/36.3/51.9, 89.7, and 28.0 versus BART 54.0/35.2/50.7, 89.5, and 27.3. The silver-label upper-bound rows are much higher (for example, CNN/DailyMail QA 55.3/31.4/51.9), exposing train/inference mismatch.

The noise ablation is a warning: lexical-NP silver-span ROUGE declines as noise is added, but the end-to-end model's CNN/DailyMail ROUGE rises from 44.8/21.0/41.6 at zero noise to 45.3/21.7/42.1 at 30% noise. Matching the quality of train-time labels to predicted test-time spans matters more than maximizing an oracle silver row.

## Design implication for `eviseq_new`

**Supported as a hypothesis:** build train-only silver evidence labels at the target-unit/source-occurrence level, retain multiple tied positives, and train a predictor whose output is used at inference. This is a concrete route for making a bridge prior query-conditioned without exposing the gold summary at test time.

**Still unproven for AFMR:** the method depends on QG/QA models, marked input spans, and shorter news datasets. EviSeq should not add inference-time evidence labels or an extra source encoder on this evidence alone. The paper's own noise result argues for confidence filtering, a predicted-prior ablation, and explicit coverage/mismatch logging.

