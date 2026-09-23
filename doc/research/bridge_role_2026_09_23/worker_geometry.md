# Bridge role audit: ordered lexical composition and source geometry

Date: 2026-09-23. Scope: research only. No source code, decoder,
grounded-copy head, data, loss, or generation path was changed.

## Finding and status correction

The XOV code in commit `efce6f8` is an **untested scaffold**, not a failed
experiment. Its existence must not be described as evidence that local
composition was rejected. The module
`CrossTokenizerOrderedValueBridge` keeps the direct projected encoder memory
for cross-attention keys and grounded copy, maps ordered decoder-token
embeddings through a rank bottleneck and depthwise `Conv1d(kernel=3)`,
reverse-scatters that local lexical signal to source rows, and adds it to
cross-attention values. The current code initializes the residual path to
zero, so its first forward function is the direct control.

That is the leading available experiment if the existing grounded-copy span
alignment tensors are in scope. It is a bridge-only memory change: decoder
and copy modules need no edits, and it does not require a new alignment/data
pipeline if it consumes the alignment already used by copy. It is not a
strict H0-only transform because its residual also reads decoder-token
embeddings. If “source hidden H0 → memory” is literal and any decoder-token
feature or alignment input is disallowed, use the geometry arm below instead.

The primary source that most directly supports the XOV role is ConvS2S
[g08](sources/g08_conv_s2s.md): its Eq. (2) uses encoder states for attention
keys and adds aligned lexical embeddings to the values. The newer Qwen3
locality study [g02](sources/g02_conv_llm.md) and the Conformer analysis
[g03](sources/g03_conformer_locality.md) support local composition as a
mechanism, but neither evaluates PPLX→Qwen nor the XOV alignment path.

## Ranked candidates

### 1. Existing XOV ordered lexical value bridge — recommend one matched run

**Current contract in `efce6f8`:** let `X=P(H0)` be the direct projected
encoder memory. For ordered decoder-token features, apply the bridge's
RMSNorm, `linear_down`, depthwise kernel-3 phrase convolution, and
`linear_up`; reverse-scatter the result with the existing copy alignment to
source rows; then form

```text
K_i       = W_K memory_norm(X_i)                  # unchanged
V_i       = W_V memory_norm(X_i + residual_i)     # lexical residual only here
copy_i    = grounded_copy(X_i)                    # unchanged
source_bias = 0
```

The implementation's zero `linear_up` makes `residual=0` at initialization.
Keep its zero-initialized identity behavior and compare it against the
direct-projection control under the same training and evaluation protocol.
Do not add a new selector, coverage state, side-memory, top-down controller,
extra loss, or decoder/copy modification.

**Why it is distinct from a cross-attention linear K/V map.** Standard
cross-attention first forms position-wise `K_i=W_K norm(X_i)` and
`V_i=W_V norm(X_i)`. XOV's residual at row `i` depends on ordered neighboring
decoder-token features and span alignment, not only on `X_i`. A fixed
per-position `W'_V X_i` has zero derivative from an unrelated aligned token
embedding or another source row to `V_i`; XOV has such a path. Thus no single
fixed K/V projection reproduces it across documents. For fixed attention
weights, XOV adds `sum_i alpha_i residual_i` to the retrieved value while
leaving key selection unchanged; this is a value-side lexical interface,
not another K projection.

**Evidence and limits.** ConvS2S supplies the closest primary precedent: its
keys are contextual encoder outputs while values add position-aligned lexical
embeddings, and its paper reports that addition as beneficial. The result is
from a different encoder/decoder and training regime. Qwen3's kernel-3 study
supports a small local inductive bias but was run in a decoder trained from
scratch. The repository's contextual-value H0 branch is not a disproof: it
pooled windows and ran region self-attention/FFN, measured
`49.657/22.098/45.920`, and lost to direct projection
`49.671/22.135/45.956`; XOV is an ordered lexical residual and has not been
measured.

**Main risks:** the cross-tokenizer lexical residual may be judged a new
side input rather than an H0 transform; alignment tensors may not be
available without prohibited data plumbing; the residual is query
independent and cannot improve which key is selected; PPLX may already carry
the local lexical information; and values/copy may overemphasize surface
tokens. Qiu et al. [g04](sources/g04_qiu_gated_attention.md) also motivate
caution about static source/value gating versus query-dependent modulation.
These are reasons to run a tightly matched ablation, not reasons to label
XOV already tried.

### 2. Masked document common-mode contrast — strict H0-only fallback

If the XOV lexical/alignment input is outside scope, the cleanest remaining
candidate is a set-conditioned channel geometry transform. It uses only the
projected source hidden state and preserves the value/copy anchor:

```text
X_i       = P(H0_i)                                  # existing projection
mu        = masked_mean_i(X_i)                        # content rows only
D_i       = X_i - mu
M_i       = X_i + g ⊙ D_i                            # key-side memory
V0_i      = X_i                                      # unchanged value/copy
source_bias = 0
```

`g` is one zero-initialized learned channel vector. Padding, prefix, and
special rows are excluded from `mu`. It needs no activation, controller,
side memory, selector, or extra objective. The operation is set-conditioned
but does not generate a per-document weight matrix.

It cannot be folded into a fixed per-token key projection: for `S` valid
rows, `dM_i/dX_j=-g/S` for `j != i`. The decoder's existing `memory_norm` is
RMSNorm and does not subtract the channel mean, so a direction-changing
common-mode contrast is not generally erased. [g06](sources/g06_whitening_geometry.md)
provides primary anisotropy evidence, [g07](sources/g07_rmsnorm.md) explains
why scale-only changes are weak, and [g01](sources/g01_transformer_attention.md)
sets the K/V algebra. This remains a conjecture for PPLX→Qwen; it has no
task-level positive result and should follow XOV only when strict H0-only
scope requires it.

**Risks:** the mean can contain useful topic or instruction signal; PPLX may
already have the right geometry; RMSNorm hides scalar scale components; and
`g` may converge to zero. A key-only change also has a narrower route to
generation improvement than a value residual.

### 3. Token-wise nonlinear channel lifting — audit only, high novelty risk

The natural nonlinear interface is `M=X+U SiLU(D RMSNorm(H0))`, with values
and copy anchored at `V0=X`. [g04](sources/g04_qiu_gated_attention.md) and
[g05](sources/g05_preprojection.md) provide primary precedents for nonlinear
interfaces around low-rank attention paths. It is a poor new contribution
here: AFMR already contains a pointwise SiLU down/up branch at
[afmr.py](/Users/kieugiangbien/Downloads/Project/LLM2Seq/src/eviseq_new/eviseq_afmr/modeling/afmr.py:261),
and its value anchor already uses the base projection at
[afmr.py](/Users/kieugiangbien/Downloads/Project/LLM2Seq/src/eviseq_new/eviseq_afmr/modeling/afmr.py:267).
Moving that same family does not establish a new bridge role. Keep it only
as an attribution check if the team needs to isolate underuse of the AFMR
branch; do not select it over the untested XOV or the strict H0 geometry arm.

## Strongest counterargument

The XOV arm may violate the literal H0-only requirement because its lexical
residual comes from decoder-token embeddings and span alignment. Even if
that plumbing is already present for copy, ConvS2S's source embedding
addition does not prove that cross-tokenizer features transfer. XOV leaves
keys unchanged, so any benefit must come from value content after retrieval;
the decoder and PPLX encoder may already provide that content. The prior
contextual-value loss, Qwen3 transfer gap, and Qiu's warning about static
value modulation all make a gain uncertain. If scope rejects the lexical
input, or a matched run shows no stable residual use and loses direct
projection, stop XOV and use the direct bridge rather than escalating to
whitening or another controller.

## Minimal falsification protocol

1. **Scope check:** verify that XOV consumes only alignment tensors already
   required by grounded copy. If it requires new extraction, span labels,
   or a new data stage, mark it out of scope.
2. **Identity check:** with the current zero initialization, verify XOV
   `values==X`, keys and grounded-copy memory equal the direct control, and
   `source_bias==0`. Confirm that no decoder or copy module is modified.
3. **Non-equivalence check:** perturb an aligned lexical token while holding
   `X` fixed and verify that the residual/value changes while `K` and copy do
   not. This directly tests the lexical value interface.
4. **Matched run:** direct projection versus XOV, same PPLX/Qwen folders,
   source span, prompt, optimizer, seed, update budget, checkpoint rule,
   deterministic decoding, copy path, and ROUGE script. Use CE only; do not
   use test results to tune the bridge.
5. **Diagnostics:** report residual RMS and cosine to `X`, key-score parity,
   cross-attention entropy, copy mass, lexical-token duplication, and ROUGE.
   For the H0 geometry fallback also report mean/variance, cosine
   concentration, and `||M-X||/||X||`; separate row scale from direction
   because RMSNorm removes row scale.

No source here guarantees a gain. The evidence supports testing XOV as an
untested, mechanically distinct lexical value interface, with the masked
common-mode transform as the only clean H0-only alternative after the
existing AFMR pointwise nonlinear branch is discounted.
