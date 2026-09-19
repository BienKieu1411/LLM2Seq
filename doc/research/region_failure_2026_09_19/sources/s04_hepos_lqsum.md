# S4/S5 — HEPOS and LQSUM

HEPOS: https://aclanthology.org/2021.naacl-main.112.pdf. The paper uses head-wise positional strides for encoder–decoder attention in long-document summarization. It motivates per-head routing, but its efficient sparse attention is a different mechanism and its results do not prove a dense AFMR bias will help.

LQSUM: https://aclanthology.org/2022.tacl-1.36.pdf. The architecture includes both a query-focused view and a query-agnostic original-document view, supporting the principle of retaining original content beside selection signals. Its task and training differ from AFMR; this is mechanism-level analogy only.
