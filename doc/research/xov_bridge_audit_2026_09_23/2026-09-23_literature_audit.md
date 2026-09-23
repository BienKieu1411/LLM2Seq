# XOV lexical value residual: bounded literature audit

Date: 2026-09-23. This audit reuses `doc/research/bridge_role_2026_09_23/debate_synthesis.md` and its source notes. It adds only targeted primary-paper checks; it does not claim a task result and does not modify `src/`.

## Decision-relevant findings

ConvS2S is the closest positive precedent. Gehring et al. compute attention from contextual encoder outputs and add the aligned source embedding to the attended value. The paper therefore supports the mechanism class “contextual keys plus point lexical values.” Its alignment is already position-preserving, so it provides no evidence that a decoder-token stream can be safely reverse-scattered into encoder positions.

Primer supports a narrower local-operator choice. Its main local operation is a width-3 depthwise convolution after each self-attention Q/K/V projection, over a contiguous sequence. The paper reports that wider depthwise or standard convolution did not help and sometimes hurt. This supports keeping the XOV pilot depthwise and width 3, but Primer does not test cross-tokenizer alignment, many-to-one pooling, post-pooling nonlinearities, grounded copy, or a value-only cross-attention residual.

The operator order remains an XOV-specific issue. Reverse-scatter pooling is linear. If convolution outputs are pooled first and SiLU is applied afterward, different decoder-token sequences can collide under the same destination weights; the existing repository probe gives one exact many-to-one collision. The constrained repair is `decoder embeddings -> RMSNorm -> lexical down -> depthwise width-3 conv -> SiLU -> reverse scatter/pool -> lexical up -> bounded residual`. This removes that particular linear cancellation. It does not make pooling injective, recover omitted-token adjacency, or guarantee order preservation.

## Initialization and redundancy counterevidence

ReZero supports an identity-preserving residual but explicitly notes that a zero residual scale makes the internal branch gradients vanish initially. A zero `lexical_up` has the same local cold-start effect for lower lexical layers even if the outer value gate is nonzero. LayerScale supplies an alternative: a small positive per-channel scale can keep the initial branch near identity while opening it without exact-zero gating. Admin adds the tradeoff that a large residual dependency destabilizes early training, while an overly light dependency can limit final capacity. These papers justify a measured initialization ablation, not a universal setting.

The redundancy case is also mixed. Tenney et al. find that BERT exposes low-level linguistic information within contextual representations and can revise lower-level decisions using higher-level context, so the encoder may already offer lexical/order cues to cross-attention. Ethayarajh finds that a static embedding explains very little variance in contextualized representations, which rejects the stronger claim that a decoder vocabulary embedding is equivalent to H0. Together they support the narrower hypothesis “XOV may make point lexical features easier to use under a finite budget,” while leaving redundancy unresolved.

## Constrained design recommendation

1. Preserve three routes in the integration contract: contextual `key_memory`, XOV-augmented `value_memory`, and the original `copy_memory`. Do not let corrected values silently replace copy features, and do not add target-derived inputs.
2. Use the corrected order above with width 3, depthwise channel-local convolution, explicit source visibility masks, and alignment-aware boundaries. Do not insert discarded decoder tokens as artificial neighbors. Keep reverse-scatter normalization explicit and report it as an alignment-weighted average.
3. Make the primary pilot identity-safe: zero output projection, existing small bounded gate, and exact forward parity at initialization. Log first-step and second-step gradients for lexical down-projection, convolution, lexical up-projection, and the gate, plus pre/post-memory-normalization residual norms. If the branch is cold-started or numerically erased, run one small-epsilon residual-scale ablation inspired by LayerScale/Admin with the same optimizer and cap.
4. Compare only matched runs: direct projection + grounded copy, corrected width-3 XOV, and (if width 3 shows a signal) a width-1/no-neighbor lexical control. A gain from width 1 supports lexical bypass access; a further gain from width 3 is required before attributing the effect to local ordered composition.
5. Select on validation CE/ROUGE under identical source visibility, updates, optimizer, clipping, precision, checkpoint rule, and deterministic decoding. Inspect actual attention-weighted value changes. No literature source supports a guaranteed ROUGE gain, and a synthetic gradient/value change is not an end-to-end result.

## Evidence boundary

The corpus has seven primary papers, but all external evidence is indirect for this repository's PPLX→Qwen summarization setup. ConvS2S validates a same-token lexical value route; Primer validates a local depthwise operator in self-attention; ReZero, LayerScale, and Admin inform residual optimization; Tenney and Ethayarajh show why encoder redundancy is plausible but not total. The recommendation is therefore to run one tightly matched, copy-anchored pilot after the existing tensor/graph checks, with no claim of efficacy until validation results exist.

## Exact URLs

- https://proceedings.mlr.press/v70/gehring17a/gehring17a.pdf
- https://arxiv.org/abs/2109.08668
- https://arxiv.org/html/2109.08668v2
- https://arxiv.org/abs/2003.04887
- https://proceedings.mlr.press/v161/bachlechner21a/bachlechner21a.pdf
- https://arxiv.org/abs/2103.17239
- https://arxiv.org/html/2103.17239
- https://arxiv.org/abs/2004.08249
- https://arxiv.org/abs/1909.00512
- https://aclanthology.org/P19-1452/
- https://aclanthology.org/P19-1452.pdf

Coordinator clarification: an expected one-step zero-up delay is not a failure. Only consider a nonzero-output initialization alternative if measured inactivity persists. A positive residual scale alone cannot open internal gradients through a still-zero output projection. The selected design retains zero output and a nonzero gate.
