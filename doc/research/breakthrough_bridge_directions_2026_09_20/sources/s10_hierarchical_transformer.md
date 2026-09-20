# Liu & Lapata 2019 — Hierarchical Transformers for Multi-Document Summarization

Primary record: https://aclanthology.org/P19-1500.pdf  
Venue: ACL 2019, pp. 5070–5081.

## Verbatim evidence

- Abstract, p. 5070: the model encodes documents “in a hierarchical manner.”
- Abstract, p. 5070: cross-document attention lets the model share information “as opposed to simply concatenating text spans.”
- Method, p. 5072: the decoder “generates a summary token by token while attending to the source input.”
- Ablation, p. 5077, Table 3: removing the global transformer layer, multi-head pooling, or paragraph position lowers all reported ROUGE scores in their setting.

## Mechanism relevant to EviSeq

Use sentence/paragraph entries only as a second memory bank, concatenated with the untouched token bank at decoder cross-attention. Give each entry a type/position embedding and a small gate; do not replace token K/V with the pooled representation. This preserves exact lexical routes while making long-range paragraph relations directly addressable.

## Training status and limitation

This is direct summarization evidence for multi-level source access, but it uses a hierarchical encoder and a different multi-document setup. It does not establish that pooled entries will improve PubMed with PPLX/Qwen. The experiment should test token-only, span-only, and token-plus-span memory with the same CE objective.
