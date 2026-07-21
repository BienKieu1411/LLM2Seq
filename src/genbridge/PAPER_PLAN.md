# GenBridge paper experiment plan

## Claim under test

A decoder-only LLM can serve as both sides of an encoder-decoder summarizer
without unmasking or continued-pretraining its backbone. A small external
adapter converts causal token states into bidirectional source memory, while
output-oriented suffix tokens form a compact summary plan. A query-dependent
dual-memory gate lets every decoder position choose separately between complete
token evidence and output planning instead of concatenating both memories under
one attention softmax.

The claim is supported only if the converted model improves over direct Qwen
full fine-tuning under the same checkpoint, data, epochs, effective batch, and
ROUGE implementation.

## Required main results

Run WikiLingua and LR-Sum with the exact HeterSumGraph ROUGE protocol:

1. direct Qwen3/Qwen3.5 full fine-tuning;
2. causal encoder-decoder conversion;
3. LaMaTE-style bidirectional token adapter;
4. full GenBridge;
5. T5Gemma baseline already stored in this repository.

The small control is Qwen3-0.6B and the first scale pilot is Qwen3.5-0.8B. Do
not spend the 2B/4B budget unless
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

## Analysis required for a paper

- ROUGE-1/2/L with bootstrap confidence intervals or paired randomization.
- Repetition rate, empty/degenerate outputs, summary length, and novel n-grams.
- Salience quality against the training-derived oracle on held-out references.
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
  matched concatenated-memory variant at 0.8B.
- Scaling gate: the 2B result preserves the gain.
- Strong-result gate: only claim superiority to T5Gemma if the same test split,
  preprocessing, generation policy, and HeterSumGraph ROUGE all match.
