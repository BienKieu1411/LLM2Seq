# GenBridge paper experiment plan

## Claim under test

A decoder-only LLM can serve as both sides of an encoder-decoder summarizer
without unmasking or continued-pretraining its backbone. A small external
adapter converts causal token states into bidirectional source memory, while
output-oriented suffix tokens form a compact summary plan. A query-dependent
dual-memory gate lets every decoder position choose separately between complete
token evidence and output planning instead of concatenating both memories under
one attention softmax.

The external token and evidence-unit attention blocks retain Qwen-style RoPE
under a full bidirectional mask. This supplies an explicit order/distance
mechanism for procedural summaries without changing the causal backbone or
adding randomly initialized absolute position embeddings. Decoder-to-source
cross-attention remains unrotated, matching the standard T5Gemma design.

To preserve pretrained generation at initialization, native Qwen token and
suffix-plan states remain as direct residuals in decoder coordinates; both
learned adapters start as gated corrections. Query and memory RMSNorms are
copied together with the native self-attention Q/K/V projections. This makes
self-attention-initialized cross-attention meaningful before full fine-tuning
instead of silently changing the pretrained projection contract.

The salience objective uses equal aggregate positive/negative class weight.
This is required by the measured WikiLingua evidence imbalance (13.2% positive,
roughly 1:6.58) and prevents the trivial all-negative content selector.

The claim is supported only if the converted model improves over direct Qwen
full fine-tuning under the same checkpoint, data, epochs, effective batch, and
ROUGE implementation.

## Required main results

Run WikiLingua and LR-Sum with the exact HeterSumGraph ROUGE protocol:

1. direct Qwen3 full fine-tuning;
2. causal encoder-decoder conversion;
3. LaMaTE-style bidirectional token adapter;
4. full GenBridge;
5. T5Gemma baseline already stored in this repository.

The small control and first architecture pilot use Qwen3-0.6B. Do not spend
the Qwen3-1.7B/4B budget unless
the full adapter beats direct Qwen and the causal-ED model on validation/test
ROUGE-2 without obvious repetition or empty-output failures.

## Required ablations

| Variant | Question |
|---|---|
| `causal_ed` | Does conversion alone help? |
| `lamate_style` | Is generic token bidirectionality sufficient? |
| `hierarchical` | Does sentence salience help without output-plan memory? |
| `no_salience_loss` | Is reference-derived content-selection supervision useful? |
| `plan_only` | Does a 16-token bottleneck lose lexical detail? |
| `concat_memory` | Is separate gated attention better than concatenating plan and evidence? |
| `layer_fusion` | Does fusing causal backbone layers justify its memory cost? |
| `adapter_2layers` | Is the four-layer main adapter better than the former two-layer setting? |
| `adapter_8layers` | Does further depth help, or only add compute/overfitting? |
| `no_adapter_rope` | Does explicit order modeling in the external bidirectional blocks improve ordered summaries/ROUGE-2? |

An embedding-initialized encoder is optional rather than a main result. Only
after the 0.6B architecture gate passes, compare a
`Qwen3-Embedding-0.6B` encoder against the matched regular `Qwen3-0.6B`
encoder. The released tokenizers currently have identical token-to-ID mappings
and special-token roles, which the loader verifies at runtime. Treat this as an
encoder-pretraining-objective ablation; both variants still use the same
external bidirectional GenBridge adapter.

## Analysis required for a paper

- ROUGE-1/2/L with bootstrap confidence intervals or paired randomization.
- Repetition rate, empty/degenerate outputs, summary length, and novel n-grams.
- Salience precision/recall/F1 and threshold-independent average precision
  against the training-derived oracle on held-out references.
- Plan-only versus full-memory examples involving names, numbers, and ordered
  procedural steps.
- Per-layer and per-position plan-gate statistics, including content words
  versus function words and early versus late decoding positions.
- Parameter count, training memory/time, and decoding speed for all scales.
- At least two datasets with different summary formats; preferably a second
  language/domain if time allows.

## Go/no-go gates

- Plumbing gate: overfit 128 samples and obtain non-degenerate generations.
- Architecture gate: full GenBridge beats direct Qwen, causal-ED, and the
  matched concatenated-memory variant at 0.6B.
- Scaling gate: the 1.7B result preserves the gain.
- Strong-result gate: only claim superiority to T5Gemma if the same test split,
  preprocessing, generation policy, and HeterSumGraph ROUGE all match.
- Direct-Qwen generation must exclude its source prompt from repetition and
  no-repeat-ngram history; GenBridge must likewise exclude its fixed decoder
  instruction. This keeps evidence copying comparable to a seq2seq decoder.
