# AdaBiMask-Qwen3.5

This folder is a standalone implementation of budgeted bidirectionalization for
low-resource summarization. It contains its own collator, generation code,
configs, and processed WikiLingua splits; no `llm2seq` sibling is required.

## Main architecture

```text
dataset-specific chat prompt
  -> Qwen/Qwen3.5-0.8B source encoder (24 layers, full fine-tuning)
       -> AdaBiMask over 6 native groups
          - full-attention layer: remove only the causal triangle
          - Gated DeltaNet layer: fuse forward and reversed-source passes
       -> identity memory bridge (both sides are width 1024)
  -> 16-layer causal Qwen3.5 decoder copied uniformly from the 24 layers
       -> pretrained native token mixer + FFN
       -> gated GQA cross-attention initialized from Qwen self-attention
       -> tied pretrained embedding / LM head
```

Qwen3.5-0.8B is hybrid rather than a conventional 24-layer Transformer. Its
layout is six repetitions of `3 x Gated DeltaNet + 1 x full attention`.
AdaBiMask therefore groups the model along these six native blocks. With the
default budget of two groups, exactly eight source layers gain right context.
The target decoder is never unmasked.

For a full-attention source layer, bidirectionalization is a padding-preserving
mask change. DeltaNet has no causal triangle to remove, so its right-context
branch evaluates the same pretrained recurrence on the reversed source and
flips the output back. Learned gates start at 0.01 and are ramped over the first
20% of full fine-tuning. This keeps the initial encoder almost exactly causal.

## Training schedule for one B200

The default configuration always runs 10 epochs:

1. epochs 1--2: freeze all pretrained Qwen weights and train only memory
   cross-attention, cross-norm, and residual gates;
2. epochs 3--10: full fine-tune encoder and decoder with differential learning
   rates (`8e-6` encoder, `1e-5` decoder, `1e-4` new cross-attention);
3. do not run validation or ROUGE between epochs; save only `final.pt` after
   the requested training schedule finishes.

The B200 config uses bf16, TF32, fused AdamW, gradient checkpointing, physical
batch 32, accumulation 1, and a 3072-token source. Installing
`flash-linear-attention` and `causal-conv1d` on the CUDA machine is recommended
for the Qwen3.5 DeltaNet fast path; the official PyTorch fallback remains
correct but is slower.

During training and validation-loss evaluation, the LM head projects only
positions whose label is not `-100`. Dense full-sequence logits are retained
when labels are absent, so autoregressive generation is unchanged while padded
target tokens incur no vocabulary-projection cost.

## Environment

The local launcher selects `/Users/kieugiangbien/bienkieu_env/bin/python` when
available and otherwise uses `python3`. On another machine, set `PYTHON_BIN`
if needed.
Qwen3.5 requires Transformers 5.14 or newer.

```bash
python -m pip install -r requirements.txt
```

## Data prompts

WikiLingua (default):

```text
Tóm tắt thành các bước hành động ngắn, đúng thứ tự; không thêm thông tin.
Văn bản:
{source}
Các bước:
```

LR-Sum (`configs/datasets/lrsum.yaml`):

```text
Tóm tắt bài báo bằng một câu nêu sự kiện chính; không thêm thông tin.
Bài báo:
{source}
Tóm tắt:
```

Both training and inference call the same tokenizer chat template and preserve
the short output label at the end of the encoder input.
Serialized wikiHow image objects are removed from both source and target at
load time, so metadata tokens cannot become generation targets.

## Training and evaluation

Upload or copy this complete `adabimask` folder, open a terminal in it, and
install the requirements. The included files are:

```text
data/processed/train.jsonl       13,999 examples
data/processed/validation.jsonl   1,680 examples
data/processed/test.jsonl         3,901 examples
```

Serialized wikiHow image metadata has already been removed from source and
target text. Runtime cleaning remains enabled as a safety check.

For a quick A100 end-to-end check (256 examples, one interface epoch and one
full-fine-tuning epoch):

```bash
./run.sh train --config configs/a100_smoke.yaml
```

The smoke run writes `runs/a100_smoke/final.pt`. It uses batch 1,
accumulation 8, a 512-token source limit, and does not change the full B200
configuration.

Run the default 0.8B model:

```bash
bash run.sh train \
  --config configs/base.yaml \
  --model-size 0.8B
```

Switch both encoder and decoder to 2B with one option. A scale override gets a
separate output directory, so this writes `runs/adabimask/base_2b/final.pt`:

```bash
bash run.sh train \
  --config configs/base.yaml \
  --model-size 2B
```

The same option applies to an ablation group:

```bash
bash run.sh ablate --group pilot --model-size 0.8B
```

Each run produces only `resolved_config.yaml`, `train.log`, and `final.pt`.
ROUGE is computed once after training with the standard `rouge-score`
tokenizer (`use_stemmer=False`):

```bash
bash run.sh eval \
  --config runs/adabimask/base/resolved_config.yaml \
  --checkpoint runs/adabimask/base/final.pt \
  --output runs/adabimask/base/test_predictions.jsonl
```

## Required comparisons

| Config | Controlled question |
|---|---|
| `direct_qwen` | Full-FT original decoder-only Qwen3.5 |
| `causal` | Does the encoder-decoder conversion alone help? |
| `full` | Does changing every source layer hurt or help? |
| `bottom/middle/top/random_k8` | Does placement matter at equal budget? |
| `learnable_k8` | Does learned placement beat fixed placement? |
| `learnable_k4/k8/k12` | Quality versus bidirectional compute |
| `learnable_k8_lora` | Full FT versus parameter-efficient encoder adaptation |

All encoder-decoder variants share the same pretrained 16-layer decoder,
cross-attention initialization, prompt, optimization schedule, and source
length. Only the source routing policy changes.

Run the dependency-free tests with `bienkieu_env`:

```bash
bash run.sh test
```
