# XOV proponent debate note

Date: 2026-09-23. Scope: research only; no source edits, training, model
downloads, or commits.

## Setup

Focus: Does the historical XOV route supply a defensible complementary
operation to decoder cross-attention, and is it worth one bounded audit on this
task? The factual question is what the route computes and what evidence
supports it. The decision question is whether that evidence justifies a pilot;
it does not justify a score or novelty claim.

Pass: Clarify -> Information -> Assumptions -> Reasoning -> Alternatives ->
Conclusion, with a peer exchange and explicit falsifiers.

## Clarify

The strongest proponent claim is conditional: XOV can expose an aligned,
ordered lexical signal through cross-attention values while leaving the base
source representation available for keys. This is a structural distinction
from a fixed per-row K/V map only when the bridge is allowed to consume the
existing aligned decoder-token IDs. It is not a claim that PPLX states lack
lexical order, that XOV adds information unavailable in principle, or that it
will improve ROUGE.

## Information

- [CITED] At efce6f8, xov.py computes X = direct_projection(H0), builds
  decoder-token embeddings through RMSNorm, rank-256 projection and a
  depthwise kernel-3 convolution, reverse-scatters them to source positions,
  and returns value_memory = X + g*delta while retaining memory = X and
  copy_memory = X. This is a verified historical contract, not an end-to-end
  result.
- [CITED] Historical decoder.py forms keys from memory_norm(memory) and values
  from memory_norm(value_memory). Thus, at the first cross-attention call for
  fixed queries, XOV can change retrieved content without changing key scores.
  Later queries can change because earlier values changed.
- [CITED] Historical model.py passes copy_memory to grounded_copy.prepare,
  separating the value experiment from the copy state. The current
  implementation lacks that third field and routes value_memory into grounded
  copy, so a faithful current port requires either accepting a coupled copy
  experiment or adding explicit plumbing.
- [CITED] Vaswani et al. define attention output as a weighted sum of values
  (sources/01_transformer.md). Gehring et al., ConvS2S section 3.3 Eq. (2),
  use contextual encoder outputs for attention scores and add source point
  embeddings to values (sources/a08_gehring_convs2s_key_value.md). This is
  direct mechanistic precedent, not transfer evidence for PPLX->Qwen.
- [CITED] Chang et al. report that local convolution plus attention can help
  while noting that self-attention can express convolution with enough heads
  (sources/a02_chang_convolution_attention.md). The appropriate claim is a
  finite-budget inductive bias, not new asymptotic expressivity.
- [CITED] The root synthetic order probe found a real limitation: with all
  four tokens aligned to one encoder position, decoder ID sequences
  [1,2,3,4] and [1,3,2,4] collide exactly under the historical
  convolution -> reverse-scatter -> SiLU order (maxdelta = 0). Moving SiLU
  before scatter gives 0.0009185 in that collision fixture, while one-to-one
  alignment is unchanged. This falsifies the broad claim that the current
  placement always preserves order, not the narrower claim that it can help on
  real alignments.
- [MISSING] No matched PPLX->Qwen XOV run, validation loss, ROUGE, or
  evidence-quality result exists. XOV is untested here; its presence in Git is
  not negative experimental evidence.

## Assumptions

- Assumption: aligned decoder-token IDs and overlap weights already used by
  grounded copy are permitted bridge inputs. — If false: XOV is outside the
  literal H_encoder->memory scope and should be rejected.
- Assumption: some lexical residual directions survive memory RMSNorm and
  decoder W_V projections. — If false: the route has no downstream effect
  even if its pre-normalization norm is nonzero.
- Assumption: a finite training budget makes an explicit lexical/local path
  easier to use than asking the existing stack to reconstruct it. — If false:
  XOV is redundant capacity and direct projection should win or tie.
- Assumption: overlap alignment retains enough local order for kernel 3 to
  matter. — If false: use identity/kernel 1 as a diagnostic or stop; do not
  call the convolution an ordered contribution.
- Value premise: a cheap, interpretable pilot is worthwhile under uncertainty.
  — If rejected: run the direct control only.

## Reasoning

For source row i, let N be the existing RMS normalization and W_V the
decoder value projection. With fixed attention weights A, direct versus XOV
value output differs by:

    A [ N(X + g R) - N(X) ] W_V

A radial component of R can be removed by RMSNorm; a direction can lie in a
W_V null space; and weighted rows can cancel. A nonzero residual tensor is only
a route diagnostic. The decisive check is a change in post-normalization,
post-W_V, and attended outputs, followed by an unchanged-copy control when the
historical contract is required.

The structural distinction from a fixed per-row map is narrower. A map
W'_V X_i cannot, for arbitrary examples, depend on another row's aligned token
or on a decoder-token ID that is not recoverable from X_i. XOV can, because its
residual is formed from an ordered token stream and reverse alignment. That is
representational non-equivalence for the expanded input interface. It does not
prove new information: a contextual encoder may already encode the same token
identities and neighborhood relations. The practical thesis is optimization
bias under fixed width, depth, and updates.

The copy argument also needs precision. Grounded copy's lexical bank is
per-token, and its pooled context comes from contextual X; it is not correct
to say grounded copy has no order. XOV's narrower distinction is an explicit
local neighbor operation on the aligned decoder-token sequence before
cross-attention values are consumed by every decoder layer. The copy head's
existing lexical route may already make the signal redundant. Current
value_memory plumbing also confounds any gain with changed copy keys, so a
current-task score cannot be attributed to cross-attention values alone unless
the historical copy_memory = X contract is restored.

The zero lexical_up initialization establishes exact direct-path identity at
step zero. It does not establish a useful learned path: first backward opens
lexical_up, while down/conv/gate gradients can remain zero until an update
makes the up projection nonzero. The bounded gate and unit cap limit damage but
do not guarantee preserved rankings or a positive objective.

Bias/fallacy scan: anchoring on ConvS2S is a risk because its model and
objective differ. Treating a synthetic route check as a task result would be a
hasty generalization. Treating "not novel in principle" and "not worth testing
under a finite budget" as the same conclusion would be a false dichotomy.

## Peer exchange: skeptic objections and proponent responses

The intended direct send_message collaboration channel was unavailable in this
runtime. The concrete exchange below records the claims sent for relay to
/root/xov_skeptic_debate and responds to the objections already surfaced in the
shared audit.

Objection 1 — H0 is contextual and already carries lexical order.

Response: conceded as a representational possibility. The proponent claim is
not recovery of missing facts. XOV adds an alignment- and decoder-tokenization-
specific path that may be easier for a finite pretrained stack to use. Test
this with aligned-ID perturbation, post-Norm/post-W_V effects, and a matched
direct control. If H0 predicts the residual or the pilot gives no stable gain,
the complementarity claim fails.

Objection 2 — RMSNorm, W_V, GQA, and attention averaging can erase the residual.

Response: conceded and made central to the falsifier. Report ||R||,
||N(X+gR)-N(X)||, post-W_V delta, attended-output delta, and cross-layer query
changes. A pre-normalization cosine or nonzero gradient is insufficient. GQA
repeats shared value heads, so a residual outside those subspaces is not useful;
if all deltas vanish, stop.

Objection 3 — Current grounded copy makes the pilot a coupled K/V/copy
experiment.

Response: conceded as an interface fact. Historical XOV's separate
copy_memory is required for a value-only attribution. If adding that field is
outside scope, label the pilot explicitly as coupled and measure copy-key
cosines, copy mass, cross-attention deltas, and CE separately. Do not claim a
pure cross-attention gain.

Objection 4 — The current convolution/scatter order loses order under overlap
collisions.

Response: the synthetic collision is real. An activation-before-scatter
variant is justified only as an attribution/control arm, with kernel 1 or
identity as the cheaper no-order control. It is not a new architecture or a
guaranteed fix. If real alignment is mostly many-to-one, withdraw the ordered
claim.

Objection 5 — Zero initialization can make XOV unused.

Response: check gradients after one update and residual use after two, rather
than interpreting expected first-step zeros as failure. If the route stays
zero, the gate collapses, or validation does not beat direct projection under
a matched budget, reject it.

## Alternatives

Direct projection plus existing grounded copy remains the strongest control.
An H0-only geometry transform satisfies a literal source-only constraint, but
its task bottleneck is unshown and RMSNorm may erase scale changes. A kernel-1
XOV arm isolates lexical bypass from local order; an
activation-before-scatter arm isolates the observed collision. These are
diagnostics around the single historical hypothesis, not new contribution
names.

## Conclusion

Judgment: XOV is worth one cheap, matched audit if aligned decoder-token inputs
are admissible and the copy route is either restored to copy_memory = X or
explicitly treated as coupled. The defensible contribution is a potential
finite-budget inductive bias: aligned lexical values with an optional local
composition path. Representational novelty is conditional on the extra lexical
input; asymptotic expressivity and ROUGE improvement are unestablished.

Recommended order: interface/identity check; post-Norm/post-W_V route-gradient
check; kernel-1 versus kernel-3 and token-order shuffle; activation-placement
collision diagnostic; then one same-seed validation pilot against direct
projection. Keep direct control if any required check fails.

The case is mechanically plausible but empirically weak: repository and
ConvS2S support the mechanism, while the exact task result remains missing.
Reverse the recommendation after a matched XOV failure, a scope finding that
alignment inputs are disallowed, an alignment audit showing order is mostly
erased, or a pilot whose apparent gain is explained by changed grounded copy.

