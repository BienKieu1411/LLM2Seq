# Shen et al. 2018 — DenseNMT

Source: https://aclanthology.org/N18-1117.pdf
Venue: NAACL-HLT 2018, pp. 1294–1303.

## Verbatim evidence (short excerpt)

- Sec. 3.2, p. 1297: “each decoder layer can get different attention values from different encoder layers.”

The paper’s DenseAtt-2 computes attention separately against multiple encoder layers and combines the resulting attention outputs. Its analysis argues that direct dense connections also provide shorter gradient routes to encoder layers.

## Mechanism

Keep the original token-level memory available. For each decoder layer, run several small cross-attention heads against separate depth or feature views, then sum their attended values. This is a dense skip interface: it does not edit source keys in-place and does not force all evidence through one pooled residual.

## Relevance and limitation

This is a plausible R1/R-L recovery route because lexical token identity remains in the base memory while higher-level views add separate attention outputs. DenseNMT was evaluated on NMT and uses a CNN backbone; the summarization transfer and Qwen/PPLX combination remain hypotheses. Memory and compute increase with the number of views.
