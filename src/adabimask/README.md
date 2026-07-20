# AdaBiMask

This folder is a clean experimental implementation of **budgeted layer-wise
bidirectionalization** for summarization. It is intentionally separate from
`src/llm2seq`; no legacy model, config, or checkpoint format is modified.

## Research question

Given a pretrained causal LLM, can a learned budget of bidirectional source
layers produce a better sequence-to-sequence summarizer than:

1. direct causal Qwen LoRA;
2. a causal Qwen encoder with the same scratch decoder;
3. full source unmasking; and
4. manually selected bottom/middle/top/random layers?

The controlled architecture is:

```text
Qwen3-0.6B-Base + LoRA
  └─ per-layer route: causal | bidirectional | learned soft mix
       └─ LayerNorm + Linear (no EncStack, fusion, salience, or global tokens)
            └─ existing 8-layer scratch decoder
                 └─ tied LM head
```

Qwen3 has 28 layers. The default policy partitions them into seven contiguous
groups of four. A budget of two groups therefore deploys exactly eight
bidirectional layers.

During learned search, causal and bidirectional attention are evaluated and
mixed by one scalar gate per group. During hard validation and after export,
top-K groups use one bidirectional attention call and every other group uses one
causal call. Dual attention is therefore a search cost, not a deployment cost.

## Folder layout

```text
src/adabimask/
  adabimask/
    mask_policy.py       # grouping, fixed controls, gate regularization
    routed_attention.py  # causal/full/soft attention routing
    encoder.py           # Qwen3-Base + LoRA wrapper
    model.py             # minimal projection + scratch decoder
    direct_baseline.py   # direct Qwen LoRA comparison
    training.py          # compact trainer
    evaluate.py          # generation + ROUGE
    export_policy.py     # soft gate -> fixed top-K config
    run_ablations.py     # experiment launcher
  configs/
    warmup_causal.yaml
    ablations/*.yaml
  tests/
```

The implementation reuses only the already-tested scratch decoder and its
cached generation routine from the sibling `src/llm2seq` project. `run.sh`
sets both Python paths automatically.

## Environment and data

The launcher automatically uses `/Users/kieugiangbien/bienkieu_env/bin/python`
when that environment exists. It can also be selected explicitly:

```bash
export PYTHON_BIN=/Users/kieugiangbien/bienkieu_env/bin/python
$PYTHON_BIN -c "import torch, transformers, peft, yaml"
```

Do not use the legacy project's pinned Transformers 4.40 environment; official
Qwen3 support requires Transformers 4.51 or newer. Install
`src/adabimask/requirements.txt` into `bienkieu_env` only if one of the imports
above is missing.

Prepare the existing WikiLingua files as JSONL:

```bash
python src/llm2seq/scripts/prepare_wikilingua_json.py \
  --input_dir src/llm2seq/datasets/wikilingua \
  --output_dir src/llm2seq/data/processed
```

## Recommended run order

### 1. Common decoder warm-up

```bash
bash src/adabimask/run.sh train \
  --config src/adabimask/configs/warmup_causal.yaml
```

This freezes Qwen and trains only the minimal projection and scratch decoder.
Every encoder-decoder ablation starts from the same checkpoint.

### 2. Falsification pilot

```bash
bash src/adabimask/run.sh ablate \
  --group pilot \
  --warmup-checkpoint runs/adabimask/warmup_causal/best.pt
```

The pilot runs causal, full, bottom-8, middle-8, and top-8. Do not spend time on
learned gates unless mask placement causes a meaningful validation difference.

### 3. Learn gates

```bash
bash src/adabimask/run.sh train \
  --config src/adabimask/configs/ablations/learnable_k8.yaml \
  --resume runs/adabimask/warmup_causal/best.pt
```

### 4. Compile and retrain a single-pass model

```bash
bash src/adabimask/run.sh export \
  --checkpoint runs/adabimask/learnable_k8/best.pt \
  --output src/adabimask/configs/compiled_k8.yaml \
  --output-dir runs/adabimask/compiled_k8

bash src/adabimask/run.sh train \
  --config src/adabimask/configs/compiled_k8.yaml \
  --resume runs/adabimask/learnable_k8/best.pt
```

The loader discards only the obsolete soft gate tensor when a learned
checkpoint is resumed into a compiled fixed policy. LoRA, projection, and
decoder weights are retained.

### 5. Direct Qwen baseline and evaluation

```bash
bash src/adabimask/run.sh train \
  --config src/adabimask/configs/ablations/direct_qwen.yaml

bash src/adabimask/run.sh eval \
  --config src/adabimask/configs/compiled_k8.yaml \
  --checkpoint runs/adabimask/compiled_k8/best.pt \
  --output runs/adabimask/compiled_k8/test_predictions.jsonl
```

## Ablation matrix

| Config | Question |
|---|---|
| `direct_qwen` | Does conversion beat the original causal summarizer? |
| `causal` | Does encoder-decoder structure alone help? |
| `full` | Is full source unmasking beneficial? |
| `bottom_k8` | Are early bidirectional layers sufficient? |
| `middle_k8` | Are middle layers the best fixed heuristic? |
| `top_k8` | Is late bidirectionality sufficient? |
| `random_k8` | Does placement matter beyond layer count? |
| `learnable_k8` | Does learned placement beat fixed placement at equal K? |
| `learnable_k4/k12` | Quality/compute curve as the budget changes |

Run `bash src/adabimask/run.sh ablate --group main --dry-run` to inspect the
commands before launching them.

Run the unit tests without adding pytest to the environment:

```bash
bash src/adabimask/run.sh test
```

## Important implementation constraints

- Source encoding always uses `use_cache=False`.
- Routed attention forces the Transformers SDPA backend. Do not change the
  encoder to `flash_attention_2`; that interface receives a different mask
  representation and cannot be routed by this wrapper.
- Padding is preserved when the causal triangle is removed.
- All fixed ablations share LoRA rank, decoder, projection, data, and training
  schedule. Only the source-layer mask policy changes.
- Checkpoints contain trainable tensors only and never store frozen Qwen base
  weights.
