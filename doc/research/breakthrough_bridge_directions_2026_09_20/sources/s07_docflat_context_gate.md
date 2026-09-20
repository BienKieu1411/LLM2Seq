# Wu et al. 2023 — Document Flattening

Source: https://aclanthology.org/2023.eacl-main.33/
Venue: EACL 2023, pp. 448–462.

## Verbatim evidence (short excerpt)

- Abstract: “NCG identifies the useful information from the distant context.”

The paper combines global attention with a neural context gate and reports improvements on document-level translation metrics. The gate is the transferable part for a separate decoder-side context branch.

## Mechanism

The transferable idea is an explicit context gate over a separate global/document branch. In this project, the gate should combine the untouched token cross-attention output and a chunk/document memory output at the decoder layer, with the gate bounded and initialized to preserve the base path.

## Relevance and limitation

DocFlat uses a document-level translation setting and its flat-batch attention is not directly suitable for the current single-document summarization batch. Only the separate context-gate principle should be reused.
