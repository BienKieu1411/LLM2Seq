# LLM2Seq Report

This folder contains the ACL-style mini-project report for LLM2Seq.

Main files:

```text
VDT_LLM2Seq.tex      # report source
VDT_LLM2Seq.pdf      # compiled report
custom.bib           # project bibliography
figures/             # PDF figures used by the report
acl.sty              # ACL style file
acl_natbib.bst       # ACL bibliography style
```

Compile from this folder:

```bash
tectonic VDT_LLM2Seq.tex --keep-logs --keep-intermediates
```

The report is written in ACL format, but it is a mini-project report rather than a conference submission. Experimental details, results, qualitative examples, limitations, and code availability are documented in the PDF.

Notes:

- Figures are stored as PDF files for sharper rendering.
- `custom.bib` contains the citations used by the report.
- LaTeX auxiliary files are build outputs and can be regenerated.
