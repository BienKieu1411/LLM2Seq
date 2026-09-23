# Independent XOV bridge audit

Date: 2026-09-23. Read-only review of `efce6f8:src/xov_bridge` and the
research design in `2026-09-23_design.md`. No production source was restored
or edited. The tensor probe is synthetic; the findings below are interface
and contract checks, not a task result.

## High-impact integration defect: the historical model is not drop-in

The historical XOV model calls its decoder as
`decoder(input_ids, memory, memory_mask, decoder_attention_mask, ...)`
(`efce6f8:src/xov_bridge/eviseq_xov/modeling/model.py:90-98`). The active
`eviseq_new` decoder has `source_bias` as its fourth positional argument and
`attention_mask` as its fifth
(`src/eviseq_new/eviseq_afmr/modeling/decoder.py:269-281`). A positional port
would therefore pass the decoder self-attention mask as a source bias and
leave the causal mask unset. This changes the computation and can fail later
when the source-bias shape is checked.

The state and copy contracts differ as well. Historical XOV returns
`memory/value_memory/copy_memory`
(`efce6f8:src/xov_bridge/eviseq_xov/modeling/outputs.py:21-28`), whereas the
active AFMR state has `memory/source_bias/controller/value_memory`
(`src/eviseq_new/eviseq_afmr/modeling/outputs.py:21-29`). Active
`grounded_copy.prepare` requires `source_bias` as its second positional
argument (`src/eviseq_new/eviseq_afmr/modeling/grounded_copy.py:41-53`). A
faithful XOV pilot needs an explicit adapter or a new state contract, named
decoder arguments, `source_bias=None` (or an explicit neutral bias), and
`copy_memory=X` passed to copy preparation. Reusing the historical call shape
is not a safe integration plan.

## Missing alignment and empty alignment are different states

Historical `CrossTokenizerOrderedValueBridge.forward` fabricates zero
alignment tensors whenever the alignment mapping is absent
(`efce6f8:src/xov_bridge/eviseq_xov/modeling/xov.py:96-108`). The model only
rejects absent alignment when grounded copy is enabled
(`efce6f8:src/xov_bridge/eviseq_xov/modeling/model.py:49-57`). Thus an XOV run
with copy disabled can silently become the direct-projection control if the
collator or integration forgets alignment fields. The design already intends
to treat an all-zero row as a valid identity fallback; preserve that behavior
only when all alignment keys are present with false masks. Reject a batch with
missing alignment keys before calling the bridge, and add a copy-disabled
regression test for this distinction.

## Boundary consistency is underspecified

The collator marks an encoder token as source content when its offset crosses
the prefix boundary (`efce6f8:src/xov_bridge/eviseq_xov/data/collate.py:139-146`),
including a token with `start < prefix_length < end`. The aligner only creates
source spans satisfying `start >= prefix_length`
(`efce6f8:src/xov_bridge/eviseq_xov/data/source_alignment.py:22-27`). A
tokenizer that merges the final prefix character with the first source
characters therefore leaves a visible key-memory row that has no lexical
value or copy alignment. Either require a tokenizer boundary at the prefix,
or clip offsets and define the content/alignment masks from the same clipped
span. Add a crossing-offset fixture; newline-terminated prompts alone do not
establish the invariant for arbitrary configurations.

## Alignment metadata must remain token-level

`copy_token_ids` has one entry per retained decoder token, while
`copy_encoder_indices`, `copy_token_indices`, and `copy_alignment_weights`
have one entry per overlap edge. `pad_source_alignments` intentionally pads
these two widths independently
(`efce6f8:src/xov_bridge/eviseq_xov/data/source_alignment.py:70-83`). The
original decoder ordinal needed by the gap-aware convolution must therefore be
a separate token-level tensor, such as `copy_token_positions`, padded like
`copy_token_ids`. It cannot be reconstructed from compact
`copy_token_indices`; discarded special or uncovered tokens have already been
removed. Repeat the position only while gathering the token stream, and keep
edge tensors for scatter. Include the position policy in the alignment
contract and test a retained-token sequence with an omitted internal token.

## Checkpoint contract gaps

The historical checkpoint spec records a generic graph name, bridge mode,
rank/kernel, two caps, and a few decoder fields
(`efce6f8:src/xov_bridge/eviseq_xov/training/checkpoint.py:23-43`). It does not
identify activation placement (`Conv -> scatter -> SiLU` versus
`Conv -> SiLU -> scatter`), the original-position adjacency rule, alignment
version/visibility policy, or encoder/decoder tokenizer and model revisions.
Those changes preserve tensor shapes, so a repaired operator can silently
load historical weights while producing a different function. Use a stable
string operator contract containing at least:

* activation placement and adjacency policy;
* overlap/row-normalization and truncation/boundary policy;
* key/value/copy memory routing;
* encoder/decoder model-config and tokenizer fingerprints.

Reject checkpoints with a missing or older contract unless an explicit
migration path is provided. The graph identifier must change when computation
changes, even if dimensions and parameter names stay the same.

There is also a direct API bug: `load_checkpoint` accepts `map_location` but
hard-codes CPU in `torch.load`
(`efce6f8:src/xov_bridge/eviseq_xov/training/checkpoint.py:122-134`). Honor
the argument or remove it. For distributed resume, `save_checkpoint` stores
only the writer's Python/NumPy/Torch/CUDA RNG states
(`...checkpoint.py:63-79`), while every rank restores those states
(`...checkpoint.py:155-161`). Save per-rank RNG states or document that DDP
resume is not bitwise reproducible; restoring rank 0's RNG on all ranks is
not an exact-resume contract.

## API guard worth carrying into the pilot

`QwenCrossDecoder.forward` returns only the final timestep when `use_cache` is
true (`efce6f8:src/xov_bridge/eviseq_xov/modeling/decoder.py:285-287`) but
still applies the full-label loss path when labels are supplied
(`...decoder.py:291-328`). `labels` plus `use_cache=True` is therefore an
unsupported combination that is not rejected and can produce a logits/label
shape error. Add an explicit guard; training should use `use_cache=False` and
generation should omit labels.

## Scope boundary

The repaired prototype's gap mask and pre-pooling activation pass the recorded
synthetic checks, including contiguous Conv1d parity, padding invariance,
empty-row identity, and delayed lower-branch gradients. Those checks do not
establish tokenizer-corpus prevalence, post-normalization value signal, CE,
ROUGE, or DDP/cache behavior. Keep the next experiment as one matched pilot
against direct projection with the explicit contracts above; no score or
complementarity claim follows from the synthetic probe.
