# Source A02 — Convolutions and Self-Attention

- Authors: Tyler A. Chang, Yifan Xu, Weijian Xu, Zhuowen Tu
- Venue: ACL-IJCNLP 2021, pp. 4322–4333
- URL: https://aclanthology.org/2021.acl-long.333/
- Type: primary peer-reviewed language-model study
- Quality: high for the stated equivalence and BERT experiments; transfer to the AFMR interface is indirect

## Verified evidence

Section 2 defines a depthwise convolution as a weighted sum of neighboring
positions in the same channel. The paper distinguishes it from standard
self-attention: attention scores use query/key representations, while a local
convolution directly assigns weights to relative positions. It derives relative
position terms as dynamic lightweight convolutions and reports that composite
convolution plus attention can improve BERT tasks.

The same paper supplies the main contrary argument. It cites a result that,
with enough self-attention heads, self-attention can express any convolution.
It therefore treats the empirical gain as an inductive-bias result rather than
proof of a new function unavailable to attention. Its own depthwise ablations
also underperform its lightweight/composite alternatives on most tasks, and
the convolutional-value rows are not uniformly better.

Short verbatim excerpts (under the source quote limit):

> “self-attention weights can express any convolution”

> “convolutions can improve self-attention by providing local position information”

## Use in this audit

The proposed bridge’s only defensible non-pointwise role is cross-position,
order-sensitive mixing before the decoder sees source keys. A per-token affine
map is different: it can be absorbed into the decoder’s trainable K/V
projections in the absence of normalization. A local operator cannot be folded
into those per-token maps for arbitrary source sequences because it depends on
neighbor rows. That distinction is structural, but this paper also prevents an
overclaim: the full encoder/decoder stack may still learn or approximate the
same computation. Local mixing should be described as an inductive bias, not
as added information or guaranteed expressivity.
