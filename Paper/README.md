# EviSeq v3 paper draft

`main.tex` is the current ACL-style draft for EviSeq v3. It describes the AFMR value-anchor bridge, copied Qwen3 cross-attention, grounded copying, four-head hierarchical semantic reading, causal prefix usage planning, and norm-preserving fusion. The draft is deliberately explicit that the revised v3 graph has no full PubMed ROUGE result yet; historical v1/v2 scores are context only. It does not describe MTP.

Compile from this folder:

```bash
tectonic main.tex --keep-logs --keep-intermediates
```

`custom.bib` contains the synchronized bibliography. The older `VDT_LLM2Seq.tex` file is retained as a historical draft and is not the source for the EviSeq v3 paper. The compiled PDF keeps the main paper to 8 content pages; references and the draft-only appendix follow on separate pages.
