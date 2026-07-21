# GenBridge

GenBridge is a standalone full-fine-tuning implementation for converting
Qwen3/Qwen3.5 decoder-only checkpoints into an encoder-decoder summarizer. It
supports `Qwen/Qwen3-0.6B`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, and
`Qwen/Qwen3.5-4B` through the same architecture and data interface.

There is no LoRA training path. After a two-epoch adapter warm-up, every
encoder, decoder, embedding, LM-head, adapter, and cross-attention parameter is
trainable.

## Architecture

```text
summary instruction + source document
                  |
        pretrained causal Qwen3/Qwen3.5
                  |-------------------------------+
           causal token states        16 learned suffix plan states
                  |                    (each has read the full source)
       2-layer bidirectional                       |
          token adapter                            |
                  |                                |
        sentence/unit pooling                      |
                  |                                |
       bidirectional unit layer                    |
                  |                                |
   reference-supervised salience  --------> plan-to-unit attention
                  |                                |
      complete token memory              summary-plan memory
                  |                                |
          independent attention              independent attention
                  +--------- query-wise gate ------+
                                  |
                    pretrained causal Qwen3/Qwen3.5 decoder
                 cross-attention after each 4-layer group
                                  |
                               summary
```

The source Qwen remains causal. Bidirectionality is implemented by the external
token adapter, so the pretrained attention mask is never altered. The suffix
planning states follow the source and therefore see the complete document under
ordinary causal attention, following the useful output-centric idea in
LLM2Vec-Gen.

The main model adds three summarization mechanisms:

1. bidirectional token evidence with sentence salience supervision derived
   automatically from the training reference using greedy ROUGE-1/2 coverage;
2. output-oriented planning tokens aligned with decoder states of the gold
   summary;
3. Plan-and-Preserve gated dual-memory attention that keeps planning and source
   grounding under separate attention normalizations.

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

References are used only to create training losses. At inference, the source
document alone produces both salience and plan states.

## Training schedule

- Epochs 1-2: freeze all pretrained weights; train only the bidirectional
  bridge, planning tokens, salience head, cross-attention, cross norms, and
  cross-residual/plan-selection gates at `1e-4`.
- Epochs 3-10: full fine-tune every parameter with one common LR inside the run.
- Save only `final.pt`; evaluate once after training.

| Profile | Physical batch | Accumulation | Effective batch | Full-FT LR | Optimizer |
|---|---:|---:|---:|---:|---|
| 0.8B | 32 | 1 | 32 | `1e-5` | fused AdamW |
| 2B | 8 | 4 | 32 | `8e-6` | fused AdamW |
| 4B | 2 | 16 | 32 | `5e-6` | AdamW 8-bit states |

The 4B profile quantizes only optimizer states to fit two fully trainable 4B
backbones and their activations on one B200. Model weights and gradients remain
BF16; this is still full fine-tuning.

## Parameter counts

Counts are obtained by instantiating the official Qwen configs on the meta
device and adding the exact bridge/cross-attention modules:

| Profile | Encoder | Decoder | Bridge + plan + cross-attention | Total |
|---|---:|---:|---:|---:|
| Qwen3 0.6B | 596,049,920 | 596,049,920 | 57,081,616 | **1,249,181,456** |
| 0.8B | 752,393,024 | 752,393,024 | 44,497,934 | **1,549,283,982** |
| 2B | 1,881,825,088 | 1,881,825,088 | 78,087,182 | **3,841,737,358** |
| 4B | 4,205,751,296 | 4,205,751,296 | 225,965,074 | **8,637,467,666** |

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

The four explicit scale configs are:

```bash
# Small Qwen3 control run
bash run.sh train --config configs/qwen3_0_6b.yaml

# First pilot
bash run.sh train --config configs/qwen35_0_8b.yaml

# Larger runs
bash run.sh train --config configs/qwen35_2b.yaml
bash run.sh train --config configs/qwen35_4b.yaml
```

Equivalent scale overrides are available for dataset configs:

```bash
bash run.sh train --config configs/datasets/wikilingua.yaml --model-size 0.8B
bash run.sh train --config configs/datasets/lrsum.yaml --model-size 2B
```

Before a paper run, the Qwen3 smoke config deliberately overfits 100 training
examples (75 interface updates + 225 full-FT updates). It also verifies that
the fixed Vietnamese decoder prefix is identical in training and inference:

```bash
bash run.sh train --config configs/smoke_100.yaml
```

Do not reuse a checkpoint trained by the earlier two-epoch smoke schedule; its
decoder was trained without this prefix and received only eight optimizer
updates.

Evaluate the final checkpoint once:

```bash
bash run.sh eval \
  --config runs/genbridge/qwen35_0_8b/resolved_config.yaml \
  --checkpoint runs/genbridge/qwen35_0_8b/final.pt \
  --output runs/genbridge/qwen35_0_8b/test_predictions.jsonl
```

ROUGE is exactly the HeterSumGraph protocol already used by the Qwen and
T5Gemma baselines in this repository: `rouge==1.0.0`, NFC normalization,
lowercase, `Rouge().get_scores(hypotheses, references, avg=True)`, and F1 x 100.

## Minimum ablation table

| Config | Purpose |
|---|---|
| `direct_qwen` | Direct decoder-only Qwen full fine-tuning baseline |
| `causal_ed` | Encoder-decoder conversion with causal token memory only |
| `lamate_style` | Generic bidirectional token adapter |
| `hierarchical` | Add sentence salience, without plan memory |
| `no_salience_loss` | Test whether reference-derived salience contributes |
| `plan_only` | Test the information loss from using only 16 plan tokens |
| `concat_memory` | Replace gated dual memory with one `[plan; token]` softmax |
| `genbridge` | Full gated token-evidence + generative-plan model |

```bash
bash run.sh ablate --group pilot --model-size 0.8B
bash run.sh ablate --group main --model-size 0.8B
```
