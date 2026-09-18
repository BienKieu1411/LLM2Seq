# S01 — Top-down and bottom-up inference

**Citation.** Bo Pang, Erik Nijkamp, Wojciech Kryscinski, Silvio Savarese, Yingbo Zhou, and Caiming Xiong (2023), *Long Document Summarization with Top-down and Bottom-up Inference*, Findings of EACL 2023.

**Primary record.** [ACL Anthology landing page](https://aclanthology.org/2023.findings-eacl.94/) · [author PDF](https://aclanthology.org/2023.findings-eacl.94.pdf). The arXiv preprint is [arXiv:2203.07586](https://arxiv.org/abs/2203.07586).

**Role.** Strongest direct external support for a region/segment-to-token top-down path.

**Short source quotation (12 words).** “Top-down correction is then applied to allow tokens to capture global context.”

## What was tested

The model first computes local token representations, pools them into segment representations, applies global self-attention to segments, and sends the segment information back to tokens through top-down token/segment cross-attention. The decoder consumes the updated token representations. This is structurally close to a region-to-token contextual-value route, although it is not the AFMR implementation.

The decisive ablation is on PubMed. With the same 1024-token local window, ROUGE F1 is:

| PubMed variant | R-1 | R-2 | R-L |
| --- | ---: | ---: | ---: |
| top-down update, cross-attention | 48.34 | 21.40 | 44.22 |
| top-down update, concatenation | 47.04 | 20.36 | 43.03 |
| no top-down update | 46.97 | 20.23 | 42.88 |

The cross-attention update is therefore +1.37 R-1, +1.17 R-2, and +1.34 R-L over the reported no-update variant in that experiment. The paper states that all of these ablations use PubMed; the broader evaluation also includes arXiv, news, conversational data, and book chapters.

## Design implication for `eviseq_new`

**Supported as a hypothesis:** a bridge can have a causal route when a coarse, globally contextual representation changes the token-level representation that the decoder reads. A bridge that only adds an unobserved source summary has weaker causal evidence than one whose output is consumed by a token-level selection/read path.

**Still unproven for AFMR:** TopDownFormer uses segment cross-attention and updates encoder token states. AFMR uses overlapping token regions, a bottleneck region stack, a bounded contextual-value residual, a focus prior, and grounded copy whose values are anchored to `H0`. The paper does not test a shared AFMR source prior, a copy marginalization loss, or the AFMR checkpoint/training budget. Its PubMed gap is evidence for the mechanism family, not a forecast of an AFMR score.

