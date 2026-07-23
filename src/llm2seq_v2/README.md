# LLM2Seq-v2

LLM2Seq-v2 keeps the original LLM2Seq research idea intact:

```text
Qwen3-Embedding-0.6B encoder
        ↓ token hidden states from several depths
stable token-wise layer fusion
        ↓
identity-residual projection
        ↓
four bidirectional full-attention adapter layers
        ↓ complete source-token memory (never a compact plan)
Qwen3-0.6B autoregressive decoder
        ↓ conventional cross-attention in every decoder layer
summary
```

The deployable graph contains one encoder, one internal adapter, and one
decoder. There is no planner, second language-model branch, merged attention,
LoRA, pseudo-summary teacher, or inference-time retrieval component.

## What changed from LLM2Seq v1

The old random eight-layer decoder is replaced by all 28 pretrained
Qwen3-0.6B decoder layers. Each layer keeps its native causal self-attention and
FFN, then receives a separate source cross-attention module. Cross Q/K/V/O and
their Q/K norms are copied exactly from that same layer's self-attention before
training. The source path has an independent softmax, preserving the mechanism
that already worked in LLM2Seq v1; this is intentionally not T5Gemma-style
merged attention.

The adapter fixes three weaknesses of v1:

1. Layer fusion has a stable last-layer prior plus a zero-initialized
   token-specific correction.
2. Because both backbones have hidden size 1024, the base projection starts as
   an identity rather than destroying pretrained coordinates with a random MLP.
3. Four independently initialized full-attention layers provide the
   bidirectional source processing. Small residual gates make the adapter begin
   near the encoder representation and grow during warm-up.

The old static global-memory tokens were removed: they were appended after the
encoder stack and therefore never became document-conditioned. Sentence
salience is instead supervised as a soft attention-logit bias over the full
memory. No source token is filtered.

Qwen3-Embedding and Qwen3 use separate tokenizers. Their hidden dimensions
match, but assuming identical tokenizer/model vocabularies is unsafe and was a
source of subtle failures in the old pipeline.

The additional decoder-heavy profile keeps the identical
Qwen3-Embedding-0.6B encoder and identical 1024-wide adapter, but replaces the
decoder with Qwen3-1.7B. A final adapter projection expands source memory from
1024 to the decoder's 2048 hidden dimensions. Keeping the adapter itself fixed
prevents the comparison from attributing a larger adapter to decoder scale.

Deployable parameter counts:

| Encoder | Decoder | Adapter | Copied cross-attention | Total |
|---|---:|---:|---:|---:|
| Embedding-0.6B | Qwen3-0.6B | 75.254M | 176.225M | 1.443B |
| Embedding-0.6B | Qwen3-1.7B | 77.353M | 352.443M | 2.746B |

The 1.7B row is deliberately decoder-heavy: the additional parameters are
allocated primarily to target-language generation rather than a larger source
adapter.

## Training

The main configuration uses:

- Phase 1: 3 epochs, freeze both pretrained backbones, train adapter and copied
  cross-attention.
- Phase 2: 12 epochs, full fine-tune encoder, adapter, decoder, LM head, and
  cross-attention.
- Physical batch 32, accumulation 1, source length 3072, BF16 autocast with
  FP32 master parameters.
- Exact `rouge==1.0.0`, `from rouge import Rouge`.

Only `last.pt` is written, atomically, after epoch 15. It contains every
encoder, adapter, decoder, LM-head, and cross-attention tensor. There is no
`best.pt`, no epoch checkpoint, and test evaluation always loads `last.pt`.
Evaluation also writes `t5gemma_gap_report.json` against the locked local
T5Gemma2-1B-1B result (39.3040/19.5308/39.2955).

## Commands

```bash
cd src/llm2seq_v2

# Uses /Users/kieugiangbien/bienkieu_env by default.
bash run.sh setup
bash run.sh test

# Real architecture check with both local/Hugging Face checkpoints.
bash run.sh check-model
bash run.sh count-params

# Seen-example plumbing/overfit test only.
bash run.sh smoke --overwrite-output-dir

# Full B200 run followed by the locked 3,901-example test of last.pt.
bash run.sh pipeline --overwrite-output-dir

# Optional larger generation model.
bash run.sh smoke-decoder-1.7b --overwrite-output-dir
bash run.sh decoder-1.7b --overwrite-output-dir
```

Evaluate an existing final checkpoint:

```bash
bash run.sh eval
bash run.sh eval --max-samples 20
```

Edit all main hyperparameters in
[`configs/base.yaml`](configs/base.yaml). The model-specific output path is in
[`configs/qwen3_0_6b.yaml`](configs/qwen3_0_6b.yaml).

## Ablations

```bash
bash run.sh ablation-no-fusion --overwrite-output-dir
bash run.sh ablation-no-bidirectional --overwrite-output-dir
bash run.sh ablation-random-cross --overwrite-output-dir
bash run.sh ablation-no-salience --overwrite-output-dir
bash run.sh ablation-sparse-cross --overwrite-output-dir
```

These isolate the revised adapter and decoder initialization without changing
the central encoder-adapter-decoder structure. `random-cross` is especially
important: it tests whether copying Qwen self-attention, rather than merely
adding more parameters, explains a gain.

## Hugging Face policy

The project calls `from_pretrained` only to load the two requested
checkpoints. It contains no `push_to_hub`, upload, repository creation, PEFT,
or Hub write command. Training artifacts remain under local `runs/`.
