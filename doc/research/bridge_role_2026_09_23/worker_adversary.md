# Adversarial bridge audit: the historical XOV route

Date: 2026-09-23. Scope: repository and literature audit only. No model,
encoder, decoder, copy head, data, loss, or source file was changed.

## Verdict

The one bridge direction that merits a controlled audit is the repository's
historical **ordered lexical value residual (XOV)**. It has not been run on
the current PPLX→Qwen task, so it is an available untested candidate rather
than a failed experiment. The claim must stay narrow: it tests whether a
source-aligned lexical sequence can improve cross-attention values after the
encoder has already produced contextual source states. It does not establish
that the route will improve ROUGE.

There is one contract problem that must be made explicit. The historical XOV
module kept three objects separate: base memory for keys, lexical-corrected
values, and base memory for grounded copy (`git show efce6f8:src/xov_bridge/
eviseq_xov/modeling/xov.py:97-134`; its contract test is at
`git show efce6f8:src/xov_bridge/tests/test_xov_contract.py:189-205`). The
current `BridgeState` has no `copy_memory` field, and
`model.py:57-65` prepares grounded copy from `value_memory` whenever it is
present. Therefore a strict bridge-only port must either let grounded copy
consume the corrected values or abandon the value-only XOV contract. Adding a
third copy tensor would be plumbing outside this task. This is a factual
interface boundary, not a hypothesis.

If preserving the existing H0 copy features is mandatory, I find no evidence
for a new value-only direction under the stated bridge-only constraint. Do not
silently add `copy_memory` or describe a K-only experiment as if it preserved
the historical XOV semantics.

## Exact candidate and existing routes

Let `H0 = P(H_final)` be the projected final encoder state. In the direct
control, `P` is identity when widths match and otherwise an orthogonally
initialized linear projection (`src/eviseq_new/eviseq_afmr/modeling/afmr.py:46-64`).
For each example, use the already-required grounded-copy alignment inputs:

```text
E       = RMSNorm(decoder_embedding(unique(copy_token_ids)))
L       = Linear_down(E)
L_order = depthwise_Conv1d_kernel3(L, along copy-token order)
R       = reverse_scatter(Linear_up(SiLU(L_order)), source positions)
V       = H0 + bounded_scalar * R
K       = cross-attention projections of H0
```

Padding, prefix/instruction positions, and unaligned destinations have zero
residual. The historical implementation uses `lexical_up` zero initialization,
so its first forward is exactly the direct value path; the repository test
checks that identity at initialization and checks that the first backward
opens the lexical route (`git show efce6f8:src/xov_bridge/eviseq_xov/modeling/
xov.py:28-45,114-134`; `git show efce6f8:src/xov_bridge/tests/test_xov_contract.py:93-106,148-170`).

For the current implementation, `K` is formed from
`memory_norm(memory)` and `V` from `memory_norm(value_memory)` when supplied
(`src/eviseq_new/eviseq_afmr/modeling/decoder.py:75-85`). The strict no-new-
plumbing test therefore passes `H0` as `memory`, `V` as `value_memory`, and
records that grounded copy also receives `V` through
`src/eviseq_new/eviseq_afmr/modeling/model.py:57-65`. The encoder, decoder
layers, grounded-copy module, and CE expression remain unchanged.

The historical module is not a new code family in the literal sense: it is
already present in git history. It is a new *untested current-task direction*.
That distinction matters for the report and for any eventual paper claim.

## What could be non-redundant

The base decoder cross-attention sees one contextual vector per source token.
The current grounded-copy head also has a lexical bank: it combines a
source-memory context with an embedding of each copied token before forming
copy keys (`src/eviseq_new/eviseq_afmr/modeling/grounded_copy.py:41-78`).
XOV adds the same already-available aligned token identities through a
different route: it makes an ordered lexical sequence, applies a local
operator to that sequence, maps it back to source positions, and exposes the
result to every decoder cross-attention value path. This is potentially
complementary to the copy head because copied lexical evidence can affect
abstractive hidden states before the vocabulary head, while grounded copy
mainly changes the final mixture logits.

There is a primary precedent for the key/value distinction. Gehring et al.,
ConvS2S, compute attention scores from contextual encoder outputs but use
`z_j + e_j` in the weighted value sum, and report that adding the point input
embedding was beneficial (2017, Section 3.3 Eq. (2),
[A08](sources/a08_gehring_convs2s_key_value.md)). That supports the mechanism,
not transfer to PPLX→Qwen.

The non-redundancy is conditional. If the residual were only a fixed
per-token map of `H0_i`, a retrained `W_V` could represent the same operation
in the no-normalization case. XOV's residual instead depends on copied token
IDs, their order, and the reverse alignment, so no fixed `W_V H0_i` can
reconstruct it for arbitrary examples that share H0 but differ in aligned
lexical inputs. This is a structural distinction. It is not proof that PPLX's
contextual states or the existing copy route fail to encode the same
information in practice.

The historical XOV tests show that keys and historical copy state stayed on
the base memory while values changed. That evidence cannot be imported as a
current result: the current `BridgeState` does not carry the historical third
memory, and the current copy path will change if `value_memory=V` is used.

## Adversarial checks

**Contextual encoder redundancy.** PPLX's final states already mix neighboring
source tokens. XOV does not create new source positions or new factual
evidence; it re-encodes token identities that may already be present in H0.
The local convolution source is the aligned copy-token stream, not a raw
source sequence, so its quality is bounded by alignment coverage and token
order. If the lexical residual is nearly predictable from H0, it is extra
capacity with no distinct signal.

**Copy redundancy and confounding.** The grounded-copy head already looks up
the same copied IDs and has a separate lexical key (`grounded_copy.py:74-78`).
Because current `encode_source` feeds `V` to copy, a score increase could come
from changing copy keys rather than from the intended cross-attention value
route. This must be measured, not hidden by calling the whole output “XOV.”

**Normalization.** Every cross-attention layer applies `memory_norm` before
its key/value projections (`decoder.py:81-84`), and grounded copy applies its
own RMS normalization (`grounded_copy.py:37-39,55-78`). A pure magnitude
increase in `R` can disappear; only its direction and interaction with
`W_V` can survive. RMSNorm is scale-only and does not subtract a mean, but
that fact does not imply that the lexical residual survives the later learned
projections. A04 records the RMSNorm boundary; inspect actual tensors.

**GQA value bottleneck.** Qwen cross-attention projects to
`num_key_value_heads` and repeats K/V across query groups
(`decoder.py:61-64,117-146`). A lexical residual that is weak in the shared
value-head subspace can have little effect even when its pre-projection norm
is large. Report residual norms before/after `memory_norm`, after `W_V`, and
after GQA repetition.

**Local-order evidence.** The 2021 depthwise-convolution study gives a
mechanism for adding local position information but also reports that fixed
depthwise variants can lose to richer alternatives ([A02](sources/a02_chang_convolution_attention.md)).
The 2026 Qwen3 study reports positive average effects for selected residual
convolution placements but benchmark-level regressions for some placements
([A01](sources/a01_tian_qwen_convolution.md)). Neither paper tests an aligned
lexical value residual around a frozen/pretrained PPLX encoder. Treat the
kernel-3 choice as a probe, not as an evidence-backed optimum.

## Cheap falsification before full training

1. **Interface and identity check.** On a tiny batch with real copy inputs,
   verify at initialization that `memory == H0`, `value_memory == H0`, and
   `R == 0` on all valid and invalid rows. Check that only `V` changes after a
   finite nonzero lexical-up perturbation; keys must remain bitwise equal.
   Record the grounded-copy state from H0 and from V. If copy changes, mark
   the strict current port as a coupled K/V/copy experiment rather than a
   historical XOV replication.

2. **Route-gradient check.** Run one teacher-forced CE backward on the same
   batch. Require finite, nonzero gradients for `lexical_up`, then after one
   update for `lexical_down`, the depthwise kernel, and the scalar. A zero
   route means the bridge is unused; adding another gate or objective is not
   justified.

3. **Order ablation.** Compare the exact route with the same lexical residual
   but an identity/no-convolution operator, and with a token-order shuffle
   that preserves the aligned token multiset and source destinations. If the
   convolution and shuffled versions are indistinguishable in `V`, logits,
   and CE, the claimed ordered role is absent. If randomizing IDs leaves the
   result unchanged, the branch is likely exploiting only a generic residual
   capacity.

4. **Copy confound measurement.** For every probe, report grounded-copy key
   cosine and norm deltas, copy probability mass, cross-attention value deltas,
   and CE separately. A gain accompanied by a large copy-state shift cannot be
   attributed to an independent cross-attention value contribution under the
   current interface.

5. **Matched pilot.** Only if the checks above show a nonzero, order-sensitive
   route, run a same-seed, same-update-budget comparison against direct
   projection and the existing AFMR/value-anchor control. Keep CE and all
   decoder/copy settings fixed. Use validation CE and paired deterministic
   ROUGE; do not tune the kernel or gate on the test set. A mixed or tiny
   change is evidence about this run, not a general bridge result.

## Final boundary

The source-backed conclusion is conditional: historical XOV supplies a
plausible complementary mechanism, and ConvS2S supplies a direct key/context
versus point/value precedent. The local repository provides no current-task
XOV score or checkpoint, and the current copy contract prevents a faithful
value-only port without additional plumbing. If the team accepts copy
consuming the transformed values, XOV is the single clean untested pilot. If
the copy anchor must remain exactly H0, the honest result is “no defensible
new bridge-only direction yet,” rather than relabeling the existing
contextual-value path or claiming that a local convolution is already proven.
