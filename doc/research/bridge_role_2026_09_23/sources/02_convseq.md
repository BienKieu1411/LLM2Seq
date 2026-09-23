# S02 — Convolutional Sequence to Sequence Learning

URL: https://proceedings.mlr.press/v70/gehring17a.html
Authors: Gehring et al. Venue/year: ICML 2017. Accessed: 2026-09-23.
Type: primary research. Credibility 5/5; recency 1/5; advocacy risk medium.
Verified excerpt, abstract: “we equip each decoder layer with a separate attention module”.

Convolutional sequence representations coexist with attention in a working
translation system. This is architectural precedent, not evidence for adding
a small bridge to a modern pretrained embedding encoder.

Full PDF verified: https://proceedings.mlr.press/v70/gehring17a/gehring17a.pdf
Section 3.3, equation (2): attention keys use contextual encoder outputs,
whereas values add source input embeddings to those outputs. Authors report
the addition beneficial. This is a particularly close precedent for a lexical
value bypass, but uses source-side embeddings and a convolutional seq2seq
model, not pretrained decoder embeddings plus cross-tokenizer alignment.
This also limits novelty claims: lexical augmentation of values is not new.
