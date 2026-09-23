# XOV bridge debate — skeptical case

Date: 2026-09-23. Scope: research only. No source code, model, data,
checkpoint, training run, or commit was changed. This note records the
skeptical side of the requested exchange with the XOV proponent and ends in a
falsifiable pilot decision.

## Opening position sent to the proponent

The historical `CrossTokenizerOrderedValueBridge` is a real, untested
candidate. The skeptical claim is narrower than “XOV cannot help”: its
computational path is different, but the repository does not yet show that it
supplies information the encoder, cross-attention stack, or grounded-copy
route cannot already use. Three objections have to be resolved before
calling it complementary.

### 1. Contextual order and copy make the information claim weak

The final encoder state already contains token order and neighboring context.
Each `CopiedCrossAttention` layer can use a query-dependent weighted sum of
per-position keys and values, then let later decoder layers compose the
retrieved states. A bridge that only applies a fixed per-position map to
`H0` is algebraically absorbable into a learned value projection in the
no-normalization case; the existing synthetic probe reports
`linear_absorption_without_norm_max_error = 4.26e-14`. XOV escapes that exact
equivalence only by consuming extra inputs: decoder-token IDs, their order,
and alignment. That is a genuine computational distinction, but it is not
evidence of a unique signal in the task distribution. If those IDs are
predictable from `H0`, the branch is extra capacity and a shortcut toward
surface copying.

The existing grounded-copy head already uses the same aligned token IDs and a
decoder embedding lookup in `lexical_key`; it combines that with aligned
source context before its copy keys. In the current `eviseq_new` model,
`encode_source` calls `grounded_copy.prepare` with
`bridge.value_memory` whenever it is present. Therefore an XOV result under
the current interface changes both cross-attention values and copy context
keys. A ROUGE or CE change cannot be attributed to the intended value route
unless a copy-anchor control is measured. The historical module kept
`copy_memory = X` separate, so using the current route is a coupled K/V/copy
experiment rather than a faithful historical value-only replication.

**Falsifier:** hold encoder states and decoder weights fixed while perturbing
aligned token IDs; measure whether the value path adds information that is
not already recoverable from `H0`, and compare (a) XOV values with copy
anchored on `H0`, (b) XOV values with current copy-on-values, and (c) direct
projection. If only (b) moves the score, the evidence is for copy coupling,
not complementary cross-attention values.

### 2. Reverse scatter can erase the claimed local order

The historical order is `lexical embedding → down projection → depthwise
Conv1d(kernel=3) → reverse scatter and row renormalization → SiLU → up
projection`. The order-sensitive signal is therefore averaged after the
linear convolution has mixed neighbors. This is not generally injective.
The actual research-only `probe_order_collision.py` gives a concrete
counterexample: decoder IDs `[1,2,3,4]` and `[1,3,2,4]`, all aligned to one
encoder row, produce exact `many_to_one_order_swap_max_delta = 0.0`; the
one-to-one path gives `0.009713...`. Moving SiLU before scatter changes the
many-to-one result to `0.0009185...`, but that is a different ordering from
the historical code and still does not establish useful corpus-level order
sensitivity.

The scatter weights are not raw character or token counts. They are first
weighted by decoder-token alignment and then renormalized by encoder row, so
one encoder row covered by three decoder subwords and another covered by one
can receive equal row-normalized influence. The compact token stream also
skips masked or unmatched entries; two tokens separated by a skipped special
or unaligned token can become artificial convolutional neighbors. Kernel 3
means three decoder subwords, not three words. Smoothing can blur names,
numbers, abbreviations, and boundary distinctions that grounded copy was
designed to preserve.

**Falsifier:** on real alignment batches, report the fraction of encoder rows
with many-to-one collisions, the effective row weights before and after
renormalization, and the residual change under aligned-token permutation.
Compare the historical `Conv → scatter → SiLU` route with an explicitly
labelled diagnostic `SiLU → scatter` route and with identity/no-convolution.
If order shuffles, identity, and convolution have indistinguishable
post-scatter values/logits, the ordered-composition claim is absent. The
diagnostic route must not be silently presented as historical XOV.

### 3. RMSNorm, W_V, and GQA can make the residual inert

For each cross-attention layer, `memory_norm` is applied to `H0` for keys and
to `value_memory` for values before `k_proj`/`v_proj`. The bridge's bounded
residual is therefore used through

```text
ΔV = W_V [RMSNorm(X + gR) − RMSNorm(X)].
```

A radial component of `R` disappears under RMSNorm: the synthetic probe's
`pure_radial_norm_delta` is only `2.38e-7`. Directional components can also be
small or cancel across the learned `W_V`, the query-weighted aggregation, and
`o_proj`. Qwen-style GQA repeats each shared projected K/V head across query
groups; repetition itself is not an eraser, but a residual that is weak in
the shared value-head subspace can still have little effect after attention.
The correct diagnostics are residual RMS/cosine before and after RMSNorm,
after `v_proj`, and the actual attended/output deltas after GQA, plus
hidden/logit deltas.

There is also an optimization risk. `lexical_up` starts at zero, so the
initial forward equals direct projection. The first backward can open only
the up-projection; `lexical_down`, depthwise convolution, and the gate need a
subsequent update. The historical synthetic probe confirms this pattern:
first-step `lexical_up` gradient is nonzero while the gate/down/conv route is
zero; on the second backward all become nonzero. That proves a finite route,
not that CE training will use it under a pretrained decoder and clipping.

**Falsifier:** require finite, nonzero bridge gradients after two optimizer
steps on teacher-forced CE and report the layerwise post-norm/post-`v_proj`
signal. If the residual is consistently radial, in a `W_V` nullspace, or
below hidden-state noise after GQA, the bridge is computationally present but
functionally inert.

## Peer-exchange status

Direct worker-to-worker delivery was unavailable: this runtime exposed no
callable `collaboration.send_message`, and the proponent's fallback queue was
rejected because the spawned thread was unloaded. The parent therefore
mediated the actual proponent reply below and confirmed its contents.

## Mediated proponent reply and skeptical response

The proponent narrowed the claim to a finite-budget inductive bias. They
conceded that `H0` may already encode lexical order, grounded copy already has
contextual order, and no information-recovery claim follows. They also
accepted possible RMSNorm/`W_V`/attention cancellation, the need for explicit
historical `copy_memory`, and activation-before-scatter as a diagnostic rather
than a guarantee. Their remaining affirmative claim is that an explicit
aligned lexical route may be easier to use than asking the pretrained stack
to reconstruct it.

I accept those concessions and agree to one pilot only: corrected-order
`SiLU → scatter → up` XOV with copy anchored on direct `X`, against matched
direct projection. Keep all data, seed, budget, optimizer, clipping, decoder,
CE, and decoding settings fixed. Train a width-1/no-convolution control only
if XOV improves, to separate lexical bypass from local order. If explicit
copy anchoring cannot be wired without changing the copy contract, label the
pilot coupled K/V/copy and do not claim value-only complementarity.

## Proponent steelman and skeptical reply

The strongest proponent case is valid in part. ConvS2S supplies a direct
precedent for contextual encoder outputs determining attention scores while
aligned point embeddings enrich values. XOV adds a cross-tokenizer ordered
sequence, a local operator, and source-position scatter; no fixed `W_V H0_i`
can reproduce arbitrary changes in aligned token IDs while holding `H0` fixed.
Because the residual is seen by every decoder cross-attention layer, it could
affect abstractive hidden states before the final vocabulary/copy mixture.
Zero initialization gives an initially exact direct control, and the
synthetic probe verifies that the route can acquire gradients.

Those points establish a plausible *computational route*, not a useful
complementary *information source*. ConvS2S used a different architecture,
tokenization, training regime, and data scale; it is a mechanism precedent,
not PPLX→Qwen evidence. The fixed-`H0` counterexample proves only that XOV is
not pointwise algebraically absorbable once extra lexical IDs are admitted.
It does not show that those IDs are missing from PPLX states, nor that
cross-attention plus multiple decoder layers cannot reconstruct their useful
parts. Likewise, every-layer exposure can amplify lexical bias and copy
surface spans rather than improve factual abstraction. Identity initialization
and finite gradients answer “is the branch open?”; they do not answer “does
the branch improve validation CE or ROUGE?”

The proponent's best response to the scatter objection is to use
`SiLU → scatter`, which restores a small many-to-one order signal in the
synthetic probe. I concede that this is a reasonable diagnostic and perhaps a
minimal implementation correction if an ordered pilot is approved. I do not
concede that it is the historical XOV contract, that `0.0009185` is useful at
task scale, or that changing the operation is free of new tokenization and
alignment confounds. Keep it as a named ablation, not a post-hoc repair of a
positive result.

## Concessions and boundary

I concede four points to the proponent:

1. XOV has not been experimentally tried on the current PPLX→Qwen task;
   presence in Git is not a failed result.
2. Its token-ID/order/alignment input makes it structurally different from a
   fixed positionwise `W_V` map of `H0`.
3. ConvS2S is a relevant primary precedent for contextual keys plus lexical
   value content, so the hypothesis is not arbitrary.
4. The repository's synthetic checks establish identity initialization,
   finite delayed gradients, and a tiny post-RMSNorm residual. They justify a
   cheap controlled pilot, not a gain claim.

I retain three non-negotiable caveats: (i) current copy consumes transformed
values, so attribution needs a copy-anchor control or the result must be
called coupled; (ii) the historical pre-scatter convolution is not reliably
order-sensitive under many-to-one alignment; and (iii) RMSNorm/W_V plus
query-weighted aggregation can erase or cancel the residual despite a
nonzero bridge tensor, while GQA sharing can limit where it is expressed.

## Falsifiable verdict

Run the identity, alignment, gradient, and order-collision checks first; these
are cheap tensor probes, not extra training arms. Then run one matched pilot:
the corrected-order `SiLU → scatter → up` variant against direct projection,
with the same seed, data, source budget, update count, clipping, decoder, CE,
and deterministic decoding. Keep historical `Conv → scatter → SiLU` as a
tensor-level audit/control. Report validation CE and paired ROUGE, but also
copy mass, copy-key cosine/norm deltas, cross-attention value deltas, residual
norms at each normalization/projection boundary, and extractive bigram
ratio/output length. Use a copy-anchored `H0` arm when plumbing permits;
otherwise call the XOV arm coupled K/V/copy and do not claim value-only
complementarity. Add a width-1/no-convolution trained arm only if the XOV
pilot wins, to separate lexical bypass from local composition.

The result that would reverse this skeptical recommendation is a stable
validation improvement over direct projection that survives the copy-anchor
comparison, has nonzero attended/output signal after GQA, and disappears
under token-order shuffle or a post-hoc no-convolution audit. A matched
failure, negligible attended signal, or order-shuffle parity should retire
the ordered-composition claim for this setup. Until such a result exists, the defensible statement is
“XOV is a plausible, mechanically distinct, untested bridge hypothesis,” not
“XOV complements cross-attention” and not “XOV improves ROUGE.”
