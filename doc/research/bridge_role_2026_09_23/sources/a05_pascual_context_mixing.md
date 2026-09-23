# Source A05 — Telling BERT’s Full Story: from Local Attention to Global Aggregation

- Authors: Damian Pascual, Gino Brunner, Roger Wattenhofer
- Venue: EACL 2021, pp. 105–124
- URL: https://aclanthology.org/2021.eacl-main.9/
- Type: primary interpretability analysis of a pretrained Transformer
- Quality: useful contrary/context evidence; it does not measure AFMR or summarization

## Verified evidence

The paper analyzes BERT attention and gradient attribution and separates local
patterns from global aggregation. It reports a mismatch between attention and
attribution caused by context mixing inside the model, while also finding some
patterns that persist across layers.

Short verbatim excerpt (under the source quote limit):

> “a significant mismatch between attention and attribution distributions”

## Use in this audit

This supports the caution that a decoder attention map alone does not prove
which source representation contains local information. It also supports the
opposing hypothesis: a deeply contextualized encoder may already carry local
composition, so a bridge convolution could be redundant. The paper gives no
evidence that PPLX final states lack phrase binding, hence local mixing remains
a falsifiable optimization/inductive-bias hypothesis rather than a factual bug.
