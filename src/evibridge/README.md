# EviBridge

EviBridge is a standalone research implementation for converting a pretrained
decoder-only LLM into an encoder-decoder **summarizer without changing its
causal source backbone**. It is summary-specific but not WikiHow-specific: the
same architecture supports procedural, news, extreme, and long-document
summarization through small dataset profiles.

The folder includes its own WikiLingua data, collator, training loop,
generation, ROUGE evaluation, tests, and controlled ablations. It does not
import anything from `llm2seq` or `adabimask`.

## Research hypothesis

Translation primarily requires dense source-target alignment. Summarization
additionally requires content selection and compression. A generic
bidirectional post-encoder therefore may not be the right interface between a
causal LLM and a summarization decoder.

EviBridge tests the following hypothesis:

> A causal LLM can remain architecturally intact and become a stronger
> summarization encoder when a lightweight bridge explicitly plans salient
> evidence before a pretrained decoder generates the summary.

This is an empirical hypothesis, not a promise that the model will beat
T5Gemma. `PAPER_PLAN.md` defines the decision gates required before making that
claim.

## Architecture

```text
dataset-specific prompt + document sentences
                    |
       unchanged causal Qwen3.5 encoder
                    |
             token hidden states
                    |
       +------------+---------------------------+
       |                                        |
full token memory                         evidence-unit pooling
                                                |
                               3-layer bidirectional unit planner
                                                |
                                   supervised salience probabilities
                                                |
                         prompt-conditioned evidence-query attention
                                                |
                                      16 evidence slots
       |                                        |
       +--------------- dual memory ------------+
                            |
           full pretrained causal Qwen3.5 decoder
               + gated pretrained cross-attention
                            |
                         summary
```

The last valid source state corresponds to the dataset-specific summary prompt
after it has causally read the document. It conditions the learned evidence
queries, so the selected information can change with the requested summary
style. The evidence-unit planner is bidirectional, but the Qwen source model is not
modified.

The decoder sees both compact evidence slots and all token states. This avoids
the irreversible information bottleneck of using only pooled embeddings while
still giving every decoder layer a short, explicit content plan.

Training minimizes

```text
L = L_summary + 0.2 L_evidence + 0.01 L_slot-diversity
```

`L_evidence` uses greedy ROUGE-1/2 oracle evidence units computed from the training
reference. References are never used at inference. The evidence oracle is
precomputed in RAM once at startup for the full run; this takes roughly one
minute on the current WikiLingua split and prevents repeated CPU search during
the ten epochs.

## Why this is not LaMaTE

| Property | LaMaTE | EviBridge |
|---|---|---|
| Task assumption | Preserve dense translation alignment | Select and compress salient content |
| External modeling | Token-level bidirectional EncStack | Hierarchical evidence-unit planner and evidence slots |
| Training signal | Translation likelihood | Summary likelihood + automatic evidence supervision |
| Decoder | NMT decoder trained from scratch | Causal decoder fully initialized from the LLM checkpoint |
| Memory passed to decoder | Adapted source token sequence | Dual memory: evidence plan + complete token sequence |
| Data regime being tested | 40M stage-1 pairs + 239k stage-2 samples | About 14k WikiLingua training examples |

The repository includes `configs/ablations/lamate_style.yaml`, which applies
layer fusion and a width/depth-matched bidirectional token post-encoder. This
makes the distinction testable rather than rhetorical.

## Training schedule

The full configuration runs exactly ten epochs and saves only `final.pt`:

1. epochs 1-2: freeze pretrained source/decoder weights; train the evidence
   bridge, decoder cross-attention, cross-norms, and residual gates;
2. epochs 3-10: full fine-tune source and decoder with differential learning
   rates;
3. calculate ROUGE once after training.

Defaults use bf16, TF32, fused AdamW, gradient checkpointing, physical batch
32, accumulation 1, source length 3072, and a complete 24-layer pretrained
decoder. The 0.8B configuration is the pilot; `configs/paper_2b.yaml` is the
quality-oriented B200 run.

## Environment

The launcher uses `/Users/kieugiangbien/bienkieu_env/bin/python` when it exists.
On Kaggle or the B200 machine, set `PYTHON_BIN` or activate the desired Python
environment.

```bash
python -m pip install -r requirements.txt
bash run.sh test
```

Installing `flash-linear-attention` and `causal-conv1d` is recommended for the
Qwen3.5 fast path. Transformers' official PyTorch fallback is correct but
slower.

## Runs

### Dataset profiles

Only the summary-format profile changes across domains; model weights and
architecture code do not contain dataset branches. Profiles are selected from
the benchmark's declared output format (single-sentence, multi-sentence, or
long-document), not tuned independently against each test set.

| Profile | Evidence unit | Oracle budget | Intended summary |
|---|---|---|---|
| `wikilingua` | sentence | number of reference steps, capped | procedural multi-step |
| `lrsum` | sentence | up to 3 selected units, capped at 6 | one-sentence Vietnamese news |
| `cnndm` | sentence | number of reference sentences | multi-sentence English news |
| `xsum` | sentence | up to 3 selected units | highly abstractive single sentence |
| `arxiv` | group of 3 sentences | number of abstract sentences | long scientific document |

Every dataset is converted to the same JSONL schema:

```json
{"id": "...", "source": "...", "target": "...", "task": "summarization"}
```

Download and prepare LR-Sum Vietnamese directly inside the folder:

```bash
bash run.sh prepare-lrsum
bash run.sh train --config configs/datasets/lrsum.yaml
```

Before a full pilot, verify that the architecture can memorize 128 examples:

```bash
bash run.sh train --config configs/overfit_128.yaml
bash run.sh eval \
  --config runs/overfit_128/resolved_config.yaml \
  --checkpoint runs/overfit_128/final.pt \
  --output runs/overfit_128/train_predictions.jsonl \
  --max-samples 128
```

0.8B decision pilot:

```bash
bash run.sh ablate --group pilot --model-size 0.8B
```

Strongest proposed 2B run:

```bash
bash run.sh train --config configs/paper_2b.yaml
bash run.sh eval \
  --config runs/evibridge/paper_2b/resolved_config.yaml \
  --checkpoint runs/evibridge/paper_2b/final.pt \
  --output runs/evibridge/paper_2b/test_predictions.jsonl
```

The generic scale override also works with the base configuration:

```bash
bash run.sh train --config configs/base.yaml --model-size 2B
```

## Controlled ablations

| Config | Question |
|---|---|
| `direct_qwen` | Is encoder-decoder conversion better than direct Qwen fine-tuning? |
| `causal_ed` | Does conversion alone help without any bidirectional module? |
| `lamate_style` | Is a generic token-level bidirectional post-encoder sufficient? |
| `hierarchical` | Does evidence-unit bidirectional reasoning help without evidence slots? |
| `no_evidence_loss` | Are slots useful without reference-derived evidence supervision? |
| `slots_only` | Does an evidence bottleneck lose details without full token memory? |
| `layer_fusion` | Does multi-layer causal representation fusion justify its memory cost? |
| `evibridge` | Full proposed architecture. |

Run the complete minimum paper table with:

```bash
bash run.sh ablate --group main --model-size 0.8B
```

ROUGE uses `rouge-score`, `use_stemmer=False`, and reports F1 for ROUGE-1,
ROUGE-2, and ROUGE-L on the unchanged test split.

## Relevant foundations

- [T5Gemma](https://arxiv.org/abs/2504.06225) shows that decoder-only to
  encoder-decoder adaptation is effective, but uses continued pretraining on
  up to two trillion tokens.
- [LaMaTE](https://aclanthology.org/2025.findings-acl.490/) uses a causal LLM,
  generic bidirectional adapter and lightweight NMT decoder for translation.
- [Pretrained Encoders for Summarization](https://aclanthology.org/D19-1387/)
  supports hierarchical inter-sentence modeling.
- [Joint Guidance Induction and Faithful Summary Generation](https://aclanthology.org/2022.findings-naacl.180/)
  demonstrates that induced guidance can improve ROUGE and factuality.
- [PROM](https://aclanthology.org/2024.lrec-main.1148/) supports auxiliary
  source-copy supervision for stable, faithful abstractive summarization.
