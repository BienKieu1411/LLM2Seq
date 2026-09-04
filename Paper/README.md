# EviSeq paper draft

`main.tex` is the current ACL-style long-paper draft for EviSeq v2. It describes the PPLX-Embed to Qwen3 composition, the single-memory evidence bridge, the DualBridge prompt-conditioned route, and attention-aligned evidence contrastive learning. It includes an explicit probabilistic and gradient-based rationale for the bridge. It does not describe MTP.

Compile from this folder:

```bash
tectonic main.tex --keep-logs --keep-intermediates
```

`custom.bib` contains the synchronized bibliography. The older `VDT_LLM2Seq.tex` file is retained as a historical draft and is not the source for the EviSeq v2 paper. The compiled PDF keeps the main paper to 8 content pages; references and the optional appendix follow on separate pages.
