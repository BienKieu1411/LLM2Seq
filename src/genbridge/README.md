# GenBridge

GenBridge is a standalone full-fine-tuning implementation for converting
Qwen3 decoder-only checkpoints into an encoder-decoder summarizer. It supports
`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, and `Qwen/Qwen3-4B` through the same
architecture and data interface.

There is no LoRA training path. After a two-epoch adapter warm-up, every
encoder, decoder, embedding, LM-head, adapter, and cross-attention parameter is
trainable.

## Architecture

```text
summary instruction + source document
                  |
             pretrained causal Qwen3
                  |-------------------------------+
           causal token states        16 learned suffix plan states
                  |                    (each has read the full source)
       4-layer bidirectional                       |
       RoPE token adapter                          |
                  |                                |
        sentence/unit pooling                      |
                  |                                |
       bidirectional RoPE unit layer               |
                  |                                |
   reference-supervised salience  --------> plan-to-unit attention
                  |                                |
 native-state + gated bidirectional      summary-plan memory
          token memory
                  |                                |
          independent attention              independent attention
                  +--------- query-wise gate ------+
                                  |
                         pretrained causal Qwen3 decoder
                 cross-attention after each 4-layer group
                                  |
                               summary
```

The source Qwen remains causal. Bidirectionality is implemented by the external
token adapter, so the pretrained attention mask is never altered. The suffix
planning states follow the source and therefore see the complete document under
ordinary causal attention, following the useful output-centric idea in
LLM2Vec-Gen.

Both source and decoder use Qwen3's official non-thinking chat format. The
decoder receives only a fixed task/language instruction before cross-attending
to source memory; this prefix is masked from the training loss and removed from
scored predictions. Compact `position_ids` make source and plan RoPE positions
invariant to left padding and batch composition.

The external token and evidence-unit blocks also use parameter-free Qwen-style
RoPE with compact positions. This matters because a generic
`TransformerEncoder` applied to projected causal states has no positional
operation of its own. Cross-attention intentionally does not rotate decoder
queries against source keys: standard T5Gemma cross-attention is likewise
position-agnostic, while RoPE remains inside encoder and decoder self-attention.
The `no_adapter_rope` ablation removes only this rotation and keeps exactly the
same learned parameter count.

The main model adds three summarization mechanisms:

1. bidirectional token evidence with sentence salience supervision derived
   automatically from the training reference using greedy ROUGE-1/2 coverage;
2. output-oriented planning tokens aligned with decoder states of the gold
   summary;
3. Plan-and-Preserve gated dual-memory attention that keeps planning and source
   grounding under separate attention normalizations.

Both the token branch and the output-oriented suffix-plan branch preserve their
native Qwen states as direct residuals. Their learned adapters begin as gated
10% corrections. Matching-size models do not apply a fresh LayerNorm to these
residuals. Each injected cross-attention also copies the native self-attention
input RMSNorm for both its query and encoder memory. Consequently, copied Q/K/V
projections receive the coordinate system they were pretrained on instead of a
fully random `hidden→512→hidden` representation.

The decoder applies the same learned Q/K/V projections to both memories but
normalizes attention over them separately. At every generated position it uses
a learned gate

```text
C = (1 - g_t) Attention(q_t, H_token) + g_t Attention(q_t, H_plan)
```

to choose source evidence versus summary planning. The gate starts at `0.1`,
so a newly initialized interface uses 90% complete token evidence and does not
force the pretrained decoder through the 16-token bottleneck. This preserves
names, numbers, and bigrams that matter for ROUGE-2 while preventing thousands
of source tokens from drowning out the plan in one shared softmax.

The training objective is

```text
L = L_summary
  + 0.2 L_salience
  + 0.1 L_plan-alignment
  + 0.01 L_plan-diversity
```

Positive and negative evidence sentences are averaged as two equally weighted
classes inside `L_salience`. This matters because only `13.2%` of the current
WikiLingua training units are positive (about `1:6.58`); ordinary unit-averaged
BCE otherwise encourages an all-negative salience head.

References are used only to create training losses. At inference, the source
document alone produces both salience and plan states.

## Training schedule

- Epochs 1-2: freeze all pretrained weights; train only the bidirectional
  bridge, planning tokens, salience head, cross-attention, cross norms, and
  cross-residual/plan-selection gates at `1e-4`.
- Epochs 3-10: full fine-tune every parameter with one common LR inside the run.
- Run teacher-forced validation after every epoch, select the lowest summary CE
  as `best.pt`, and retain the terminal weights as `last.pt`. No per-epoch
  checkpoint history is kept.

The full stage and T5Gemma baseline both use AdamW
`betas=(0.9, 0.95)`, `eps=1e-8`, weight decay `0.01`, 5% linear warm-up,
and cosine decay to zero. This prevents optimizer defaults from confounding the
architecture comparison. GenBridge alone has the preceding interface warm-up.

Training uses shuffled length mega-buckets while preserving the requested
physical/effective batches. On the included WikiLingua train split this reduces
the measured token-padding factor from `2.72x` to `1.18x` and the padded
quadratic-attention proxy from `5.69x` to `1.43x` compared with fully random
batching.

| Profile | Physical batch | Accumulation | Effective batch | Full-FT LR | Optimizer |
|---|---:|---:|---:|---:|---|
| 0.6B | 32 | 1 | 32 | `1e-5` | fused AdamW |
| 1.7B | 8 | 4 | 32 | `8e-6` | fused AdamW |
| 4B | 2 | 16 | 32 | `5e-6` | AdamW 8-bit states |

The 4B profile quantizes only optimizer states to fit two fully trainable 4B
backbones and their activations on one B200. All profiles retain FP32
master/model parameters and gradients so `5e-6`–`1e-5` updates are not rounded
away; forward/backward matrix compute uses BF16 autocast. Evaluation casts the
selected best checkpoint to BF16.

## Parameter counts

Counts are obtained by instantiating the official Qwen configs on the meta
device and adding the exact bridge/cross-attention modules:

| Profile | Encoder | Decoder | Bridge + plan + cross-attention | Total |
|---|---:|---:|---:|---:|
| Qwen3 0.6B | 596,049,920 | 596,049,920 | 63,391,506 | **1,255,491,346** |
| Qwen3 1.7B | 1,720,574,976 | 1,720,574,976 | 109,570,834 | **3,550,720,786** |
| Qwen3 4B | 4,022,468,096 | 4,022,468,096 | 258,505,494 | **8,303,441,686** |

Reproduce the count without downloading model weights:

```bash
bash run.sh count-params
```

## Environment and data

The launcher uses `/Users/kieugiangbien/bienkieu_env/bin/python` when available.
On the B200 machine, activate the desired environment or set `PYTHON_BIN`.

```bash
python -m pip install -r requirements.txt
bash run.sh test
```

If an existing GPU environment raises `cannot import name 'Unpack' from
typing_extensions`, upgrade that package and restart the Python process before
running GenBridge:

```bash
python -m pip install --upgrade "typing_extensions>=4.12.2,<5"
```

Cleaned WikiLingua files are already included under `data/wikilingua/` using
the common JSONL schema:

```json
{"id":"...","source":"...","target":"...","task":"summarization"}
```

Expected files are `train.jsonl`, `validation.jsonl`, and `test.jsonl`.
Rebuild them from the existing local WikiLingua JSON files with:

```bash
bash run.sh prepare-wikilingua --input-dir /path/to/wikilingua-json
```

LR-Sum can be prepared directly:

```bash
bash run.sh prepare-lrsum
```

## Running

The one-command paper pipeline trains once and evaluates both checkpoint roles
on the identical test set:

```bash
bash run.sh pipeline --config configs/qwen3_0_6b.yaml
```

Use `--overwrite-output-dir` only for an intentional rerun. The pipeline writes
`best_test_predictions.jsonl`/`.metrics.json` and
`last_test_predictions.jsonl`/`.metrics.json`. The paper result must use
`best.pt`, selected solely by validation CE; `last.pt` is reported only as an
overfitting diagnostic and must not be selected using test ROUGE.

The symmetric scale configs are:

```bash
# Small Qwen3 control run
bash run.sh train --config configs/qwen3_0_6b.yaml

# Larger runs
bash run.sh train --config configs/qwen3_1_7b.yaml
bash run.sh train --config configs/qwen3_4b.yaml
```

Training refuses to start when the selected output directory still contains
`best.pt`, `last.pt`, legacy `final.pt`, or an incomplete-run marker. This prevents a crashed rerun from
being mistaken for a new result. To intentionally rerun the exact same config,
pass `--overwrite-output-dir`; the old checkpoints are removed before any
new training begins:

```bash
bash run.sh train --config configs/qwen3_0_6b.yaml --overwrite-output-dir
```

Two controlled capacity-allocation configs use the identical Qwen3 tokenizer
but different encoder/decoder widths:

```bash
# Recommended asymmetric run: stronger source encoder, cheaper generator
bash run.sh train --config configs/qwen3_enc1_7b_dec0_6b.yaml

# Reverse allocation with nearly the same backbone parameter budget
bash run.sh train --config configs/qwen3_enc0_6b_dec1_7b.yaml
```

The decoder tokenizer is canonical because its IDs index the decoder LM head.
Before loading a mixed model, the pipeline compares the complete encoder and
decoder token-to-ID dictionaries plus BOS/EOS/PAD IDs. A same-size but reordered
vocabulary is rejected.

Equivalent scale overrides are available for dataset configs:

```bash
bash run.sh train --config configs/datasets/wikilingua.yaml --model-size 0.6B
bash run.sh train --config configs/datasets/lrsum.yaml --model-size 1.7B
```

Before a paper run, the Qwen3 smoke config deliberately overfits 100 training
examples (125 interface updates + 625 full-FT updates). It also verifies that
the fixed Vietnamese decoder prefix is identical in training and inference:

```bash
bash run.sh train --config configs/smoke_100.yaml

bash run.sh eval \
  --config runs/genbridge/qwen3_0_6b_overfit_100/resolved_config.yaml \
  --checkpoint runs/genbridge/qwen3_0_6b_overfit_100/last.pt \
  --output runs/genbridge/qwen3_0_6b_overfit_100/predictions.jsonl \
  --max-samples 100

bash run.sh check-smoke \
  --metrics runs/genbridge/qwen3_0_6b_overfit_100/predictions.metrics.json
```

The smoke check deliberately uses `last.pt`, because its purpose is to verify
that the optimization path can overfit the 100 seen examples. Paper evaluation
uses validation-selected `best.pt`. The final command is a plumbing gate: it rejects empty/repetitive output,
common-prefix collapse, an ignored encoder, or very low overfit ROUGE. It is
not a substitute for held-out paper evaluation.

Use the dedicated mixed-width smoke config before launching the asymmetric
paper run:

```bash
bash run.sh train --config configs/smoke_mixed_enc1_7b_dec0_6b_100.yaml
```

The symmetric and mixed smoke runs write to separate `*_overfit_100`
directories to avoid stale checkpoints. Do not reuse a checkpoint trained by
an earlier smoke schedule.

Evaluate the validation-selected checkpoint once on the test set:

```bash
bash run.sh eval \
  --config runs/genbridge/qwen3_0_6b/resolved_config.yaml \
  --checkpoint runs/genbridge/qwen3_0_6b/best.pt \
  --output runs/genbridge/qwen3_0_6b/test_predictions.jsonl
```

ROUGE is exactly the HeterSumGraph protocol already used by the Qwen and
T5Gemma baselines in this repository: `rouge==1.0.0`, NFC normalization,
lowercase, `Rouge().get_scores(hypotheses, references, avg=True)`, and F1 x 100.
The metrics file also reports empty/short/long-output rates, repeated trigrams,
summary length, compression ratio, and held-out salience precision/recall/F1
plus average precision. It also records end-to-end encoder+decoder latency,
throughput, generated-token counts, and peak inference memory at the same
WikiLingua evaluation batch (`8`) as T5Gemma. Held-out references are used only after generation to
score the evidence ranking; they are never model inputs. For the direct-Qwen baseline,
repetition and no-repeat-ngram processors intentionally ignore the source
prompt and inspect generated summary tokens only; otherwise copying grounded
names and bigrams from a long document would be penalized unfairly.

For the dual-memory claim, evaluation stores plan-gate usage per injected
decoder layer, per generation step, over the first 16 output positions, and
over later positions. These are aggregated only over examples still active at
that step, so early-finished sequences do not contaminate late-position gates.

`data.max_target_length` is the generated-summary budget for every baseline;
the fixed GenBridge decoder instruction is conditioning context and does not
consume those 384 positions. After evaluating both systems, run the paired
paper comparison:

```bash
bash run.sh compare \
  --candidate runs/genbridge/qwen3_0_6b/best_test_predictions.jsonl \
  --baseline ../T5Gemma/eval_outputs/full_test/predictions.jsonl \
  --output runs/comparisons/genbridge_vs_t5gemma.json
```

This command verifies identical ID sets and exact references before reporting
ROUGE-1/2/L, 95% paired-bootstrap intervals, delta intervals, and paired
approximate-randomization p-values. It refuses an unpaired or mismatched test
set rather than producing an invalid superiority claim.

## Minimum ablation table

| Config | Purpose |
|---|---|
| `direct_qwen` | Direct decoder-only Qwen full fine-tuning baseline |
| `causal_ed` | Encoder-decoder conversion with causal token memory only |
| `lamate_style` | Generic bidirectional token adapter |
| `hierarchical` | Add sentence salience, without plan memory |
| `no_salience_loss` | Test whether reference-derived salience contributes |
| `no_memory_curriculum` | Remove plan-only training batches |
| `no_salience_attention_bias` | Remove the soft salience prior from token cross-attention |
| `no_plan_evidence_alignment` | Remove ordered plan-to-evidence supervision |
| `plan_only` | Test the information loss from using only 16 plan tokens |
| `concat_memory` | Replace gated dual memory with one `[plan; token]` softmax |
| `adapter_2layers` | Reduce the bidirectional adapter from four to two layers |
| `adapter_8layers` | Test whether doubling adapter depth adds useful capacity |
| `no_adapter_rope` | Remove RoPE while retaining the same adapter parameters |
| `genbridge` | Full gated token-evidence + generative-plan model |

```bash
bash run.sh ablate --group pilot --model-size 0.6B --evaluate
bash run.sh ablate --group main --model-size 0.6B --evaluate
```

If training already finished, generate/score the same checkpoints without
touching their weights using `--eval-only`. An intentional complete rerun needs
`--overwrite-output-dir`, matching the single-run stale-checkpoint guard.

## Why the main encoder is not Qwen3-Embedding

Qwen3-Embedding is a useful retrieval checkpoint, but it is still a causal
Qwen3 model trained to make a strong pooled last-token vector. GenBridge feeds
every token state to bidirectional evidence processing and decoder
cross-attention, so sentence-level embedding quality does not guarantee better
token-level source memory. It also does not remove the need for the
bidirectional adapter.

The regular Qwen3 model config reserves more embedding/LM-head rows than the
number of entries returned by its tokenizer. Direct tokenizer inspection shows
that Qwen3-Embedding-0.6B and regular Qwen3 currently share the same 151,669
token-to-ID entries and BOS/EOS/PAD roles. They are therefore technically
compatible with one shared tokenizer; the differing model-config vocabulary
sizes alone are not an incompatibility. The loader checks the actual tokenizer
dictionaries instead of assuming compatibility from model names or sizes.

For the paper, use ordinary Qwen3 on both sides as the main model. If the 0.6B
main result is viable, a later controlled ablation may directly compare
`Qwen3-Embedding-0.6B -> Qwen3-0.6B` against the regular 0.6B encoder. This is
an encoder-objective ablation, not a bidirectionality ablation: the external
GenBridge adapter remains responsible for bidirectional token mixing. The
ready-to-run optional config is:

```bash
bash run.sh train --config configs/qwen3_embedding_enc0_6b_dec0_6b.yaml
```
