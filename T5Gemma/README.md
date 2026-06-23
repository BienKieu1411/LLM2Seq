# T5Gemma 2 WikiLingua LoRA Baseline

Baseline fine-tuning for WikiLingua summarization with a T5Gemma encoder-decoder
model on a single L40 45GB GPU.

This folder is standalone. Upload the whole `T5Gemma/` directory to the server;
do not rely on files outside this directory.

## Model Choice

The default model is:

```text
google/t5gemma-2-1b-1b
```

This is T5Gemma 2 pretrained with a 1B encoder and a 1B decoder. It is larger
than the earlier "512M-512M" idea, but it is the checkpoint requested for this
baseline and remains an encoder-decoder text-to-text model suitable for
summarization.

Gemma-family checkpoints are gated on Hugging Face. Before running, accept the
license for the model on Hugging Face and put your token in `env.txt`.

## Folder Layout

```text
T5Gemma/
  configs/lora_l40_512.yaml
  scripts/prepare_wikilingua_json.py
  scripts/train_lora.py
  scripts/evaluate_full_test.py
  scripts/load_env.sh
  scripts/train.sh
  scripts/evaluate.sh
  install_deps.sh
  run_pipeline.sh
  smoke_check.sh
  env.example.txt
  wikilingua/
```

Expected raw data:

```text
T5Gemma/wikilingua/train.json
T5Gemma/wikilingua/val.json
T5Gemma/wikilingua/test.json
```

The converter supports WikiLingua files saved either as a JSON list or as
JSONL-style consecutive JSON objects. `src` sentence lists are joined with
newlines; `tgt` sentence lists are joined with spaces so the summary becomes one
natural paragraph.

## Server Commands

```bash
cd /path/containing
bash T5Gemma/smoke_check.sh
bash T5Gemma/install_deps.sh
cp T5Gemma/env.example.txt T5Gemma/env.txt
nano T5Gemma/env.txt
bash T5Gemma/run_pipeline.sh
```

If shell execution permission is preserved, these also work:

```bash
./T5Gemma/smoke_check.sh
./T5Gemma/install_deps.sh
./T5Gemma/run_pipeline.sh
```

## What Gets Saved

Training saves LoRA adapter-only checkpoints, not the base T5Gemma weights:

```text
runs/t5gemma2_1b_1b_lora_wikilingua/best_adapter/
runs/t5gemma2_1b_1b_lora_wikilingua/final_adapter/
runs/t5gemma2_1b_1b_lora_wikilingua/epochs/epoch_001_adapter/
...
```

If `HF_TOKEN` and `HF_REPO_ID` are set, the script pushes adapter checkpoints to:

```text
checkpoints/t5gemma2_1b_1b_lora_wikilingua/
```

Evaluation saves full-test metrics and every prediction:

```text
T5Gemma/eval_outputs/full_test/metrics.json
T5Gemma/eval_outputs/full_test/predictions.jsonl
T5Gemma/eval_outputs/full_test/eval_run_info.json
```

Evaluation outputs are pushed to Hugging Face too when enabled in the config.

## L40 45GB Defaults

The default config uses:

```text
source length: 512 tokens
target length: 512 tokens
epochs:        4
LoRA r/alpha:  16 / 32
train batch:   4
grad accum:    4
effective bs:  16
lr:            1e-4
precision:     bf16
```

If OOM appears, change:

```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
```

If VRAM is still loose and GPU utilization is good, try:

```yaml
training:
  per_device_train_batch_size: 6
  gradient_accumulation_steps: 3
```

## Manual Steps

Prepare data only:

```bash
python3 T5Gemma/scripts/prepare_wikilingua_json.py \
  --input_dir T5Gemma/wikilingua \
  --output_dir T5Gemma/data/processed
```

Train only:

```bash
bash T5Gemma/scripts/train.sh
```

Evaluate only:

```bash
bash T5Gemma/scripts/evaluate.sh
```
