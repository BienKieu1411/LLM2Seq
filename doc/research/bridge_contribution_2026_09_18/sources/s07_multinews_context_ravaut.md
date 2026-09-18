# S07 — Hierarchical inference can hurt Multi-News

**Citation.** Mathieu Ravaut, Aixin Sun, Nancy Chen, and Shafiq Joty (2024), *On Context Utilization in Summarization with Large Language Models*, ACL 2024.

**Primary record.** [ACL Anthology landing page](https://aclanthology.org/2024.acl-long.153/) · [author PDF](https://aclanthology.org/2024.acl-long.153.pdf). Code/data are linked at [MiddleSum](https://github.com/ntunlp/MiddleSum).

**Role.** Negative transfer evidence: hierarchical processing is domain-dependent and can damage a multi-document news task.

**Short source quotation (9 words).** “Yet, they seem to harm summaries on the other domains.”

## What was tested

The study evaluates six LLMs on ten datasets with five metrics and introduces MiddleSum, 225 examples from ArXiv, PubMed, GovReport, SummScreenFD, and Multi-News. Hierarchical inference splits inputs into approximately 1,500-word blocks, summarizes each block, then summarizes the concatenated intermediate summaries. This is an inference pipeline rather than AFMR training, but it isolates a useful stress case.

For Llama-2-7B on MiddleSum, ROUGE-2 changes from standard to hierarchical as follows: ArXiv 12.62→14.63, PubMed 10.97→13.36, GovReport 13.26→13.51, SummScreenFD 4.07→4.85, and Multi-News 10.43→7.06. For Llama-2-13B, Multi-News changes 10.38→6.71 and GovReport 13.56→10.21. These results show that a hierarchy can help scientific subsets while hurting Multi-News, with the direction depending on model and domain.

## Design implication for `eviseq_new`

**Supported:** include Multi-News or another multi-document stress slice, and report domain-stratified evidence/faithfulness metrics. A bridge that helps PubMed alone should not be called generally useful.

**Still unproven:** the models, prompts, sampling, and inference composition differ from supervised AFMR. The numbers do not predict AFMR's bridge effect; they only rule out universal claims about hierarchical processing.

