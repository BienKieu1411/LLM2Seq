# Does XOV complement cross-attention?

Design follow-up (2026-09-23): [XOV reviewed and repaired design](../xov_bridge_audit_2026_09_23/2026-09-23_design.md)
specifies activation-before-pooling, original-token adjacency masks, explicit copy anchoring and checkpoint contracts.
The historical audit below remains evidence/provenance; the linked design governs the proposed next implementation.

Date: 2026-09-23. Research and synthetic probes only; no production source edits.
XOV is untested on the user's task, explicitly confirmed by the user.

## Decision boundary

XOV provides a distinct computational route into cross-attention values, but
neither its existence nor the published precedents prove useful complementary
information on this task. The encoder already contextualizes lexical order,
and grounded copy already uses decoder lexical embeddings. The defensible
hypothesis is easier access to ordered lexical features under a finite training
budget, not information absent from the entire baseline or a capability that
cross-attention can never learn. No positive ROUGE forecast is established.

For fixed queries and key memory at a layer, its exact intervention is

    delta_context = A * [N(X + g R) - N(X)] * W_V

where N is memory RMSNorm and A is the baseline attention distribution.
Changing values need not change retrieval at that same fixed-query layer.
Subsequent queries, later-layer attention and copy probabilities can change.
The residual may disappear through row normalization, the value projection,
or cancellation in the weighted sum. Nonzero gradients/residuals alone do not
show improved summaries. The encoder and decoder can also adapt to learn
similar features without XOV; universal impossibility claims are inappropriate.

## Concrete weakness found in historical code

Historical XOV at efce6f8 uses this ordering:

    embeddings -> down -> linear depthwise conv -> reverse scatter/mean
               -> SiLU -> up -> bounded residual into values

Four decoder subwords mapping wholly into one encoder token produce equal
reverse-scatter weights. With the first/last subwords fixed, swapping the two
interior subwords leaves the sum of width-3 linear convolution outputs
unchanged: every interior input has the same total coefficient across the
outputs. Applying SiLU after the mean cannot recover that lost distinction.
This is a counterexample to general order preservation, not evidence that all
examples lose order. Encoder H0 may still distinguish the two source strings.

`probe_order_collision.py` instantiates the actual historical bridge class
with tiny random embeddings and nonzero output projection. It holds H0 fixed
to isolate the added branch; this is a mechanistic intervention, not a natural
counterfactual about an encoder on real text. Results:

| Probe | Maximum value-memory difference |
|---|---:|
| Historical order, many-to-one, internal token swap | 0.0 |
| Historical order, one-to-one, same token swap | 0.0097132623 |
| SiLU before scatter, many-to-one, same swap | 0.0009185374 |
| Original vs changed order on one-to-one control | 0.0 |

The research-only alternative computes SiLU immediately after convolution,
before averaging onto encoder positions. It removes this specific linear
cancellation and preserves the one-to-one control. It does not make pooling
injective or guarantee all token orders survive; it changes the many-to-one
operator and needs its own training validation. No src code was changed.

## Other conditions before interpreting a result

- Preserve explicit key_memory/value_memory/copy_memory routing. Current
  eviseq_new would otherwise supply corrected values to grounded copy.
- Keep source visibility identical; no extra tail tokens or target-derived
  features in XOV. Convolution uses decoder subwords, not words.
- Check retained-token adjacency: alignment filtering may omit internal
  tokens and create artificial neighbors in the compact stream. Measure its
  prevalence before choosing a boundary-handling change.
- Reverse-copy weights are normalized by decoder token and then renormalized
  at encoder destinations. They are not raw character-overlap proportions;
  describe the choice accurately and do not label it a proven bug.
- At zero-initialized output projection, first-step gradients into the lower
  lexical layers are zero by design. The next backward opens those routes.
- Copy memory fixed at inference does not imply training isolation: decoder
  embeddings are shared and receive an additional gradient route from XOV.
- A perturbation of token IDs also changes copy lexical keys unless that
  intervention is isolated to the bridge. Do not claim copy-state parity from
  only unchanged copy_memory when lexical IDs/embeddings changed.

## Decision-relevant validation

Start with the existing matched direct_projection + grounded_copy control.
If run settings differ, aggregate scores cannot attribute a bridge effect.
For an XOV pilot, preserve data, visible source tokens, prompt, global batch,
updates, optimizer/clipping, precision, checkpoint rule and deterministic
decoding. Use validation for selection; CE only.

Measure actual attended value/output changes, teacher-forced CE and validation
ROUGE. Check gradients and order sensitivity as prerequisites, not success
metrics. If a trained XOV branch is disabled at evaluation, deterioration
shows model reliance, not superiority over a separately trained direct control.
Neither a failed short pilot nor a small numerical benefit proves long-run
failure/success. Report uncertainty rather than selecting arbitrary gates.

If XOV gains, use a no-neighbor/width-1 lexical control to distinguish the
lexical bypass contribution from local-order composition. Match capacity where
practical. A gain from lexical bypass alone does not substantiate an ordered-
composition claim. Do not increase copy bias, generation length, or decoding
search to manufacture a bridge ablation gain.

## Paper evidence

ConvS2S §3.3 Eq2 uses contextual keys with values augmented by source embeddings:
https://proceedings.mlr.press/v70/gehring17a/gehring17a.pdf
It supports a lexical value route, but neither a unique novelty claim nor an
expected improvement for pretrained PPLX/Qwen long-document summarization.
Primer adds convolution near Q/K/V in self-attention:
https://arxiv.org/abs/2109.08668
Its intervention and training setup differ. These are precedents, not a task
effect estimate. See per-source records and the earlier decision report.

## Agent exchange

Two luna_worker agents completed their reviews:

- `debate_proponent.md`: contextual keys with an ordered lexical value route
  can offer a finite-budget inductive bias. The proponent concedes that H0
  and contextual copy may already encode the same useful information, that
  gradients are insufficient, and that scatter/norm/projection can erase it.
- `debate_skeptic.md`: a different computational route is not evidence of
  extra useful information. The skeptic concedes the route is structurally
  distinct, has a relevant ConvS2S precedent, and is untested rather than failed.

Their direct messaging tool was unavailable. The root read both actual drafts,
sent the skeptic's objections to the proponent, and sent the proponent's actual
concessions to the skeptic. Both responded to the root-mediated exchange.
The skeptic's final reply explicitly accepts one corrected-order, copy-anchored
pilot against matched direct projection. The proponent's final reply accepts
the same pilot with no gain/novelty claim. This is agreement on an experiment,
not independent empirical confirmation of the architecture.

Coordinator decision: do not run the historical ordering unchanged if the
purpose is to test ordered lexical composition. Use `conv -> SiLU -> scatter
-> up` with explicit base copy memory. Keep the historical route as a cheap
tensor reference, not an additional full training run. Use the existing direct
run only if settings are actually matched; train width-1 only after a gain.

Corrections to any stronger wording in individual notes: GQA head repetition
alone cannot erase a tensor; measure projected and attention-weighted outputs.
Changing only value_memory through current copy plumbing couples values and
copy context, not key-memory tensors. A shuffled-token or disabled-branch
evaluation of a trained model measures reliance/perturbation sensitivity,
not the benefit relative to retraining a matched control. Many-to-one alignment
frequency alone does not prove all order is lost; the collision probe establishes
one case only. This synthesis governs the final interpretation.
