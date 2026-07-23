# T5Gemma Full Fine-tuning Baseline

Full-parameter fine-tuning of `google/t5gemma-2-1b-1b` and
`google/t5gemma-2-4b-4b` for WikiLingua or
Vietnamese LR-Sum summarization. This pipeline does not use LoRA or PEFT: the
encoder, decoder, embeddings, and language-model head are all updated.

Evaluation uses the same HeterSumGraph-compatible `rouge==1.0.0` protocol as
the other project models: NFC normalization, lowercase text, stored whitespace
tokenization, and macro-averaged F1 on the 0--100 scale.

The folder is standalone. Gemma-family checkpoints are gated on Hugging Face,
so accept the model license and configure `HF_TOKEN` for downloading the base
model. All trained checkpoints and evaluation outputs remain local; upload
paths are hard-disabled.

## Main files

```text
T5Gemma/
  configs/wikilingua_full_3072.yaml
  configs/wikilingua_full_4b_4b_3072.yaml
  configs/lrsum_full.yaml
  scripts/prepare_wikilingua_json.py
  scripts/prepare_lrsum_json.py
  scripts/train_full.py
  scripts/evaluate_full_test.py
  run_pipeline.sh
  run_4b_pipeline.sh
  run_lrsum_pipeline.sh
  smoke_check.sh
```

## Run WikiLingua

```bash
cd /path/containing/src
bash T5Gemma/smoke_check.sh
bash T5Gemma/install_deps.sh
cp T5Gemma/env.example.txt T5Gemma/env.txt  # only if env.txt does not exist
bash T5Gemma/run_pipeline.sh
```

The larger single-B200 baseline has an isolated output directory and starts
with physical batch 1, gradient accumulation 32, gradient checkpointing, and
FP32 AdamW/master parameters with BF16 autocast:

```bash
bash T5Gemma/run_4b_pipeline.sh
```

Expected raw WikiLingua files are
`T5Gemma/datasets/wikilingua/{train,val,test}.json`.
The converter accepts a JSON list or JSONL-style consecutive objects.

## Run LR-Sum

```bash
cd /path/containing/src
LRSUM_CONFIG=T5Gemma/configs/lrsum_full.yaml \
  bash T5Gemma/run_lrsum_pipeline.sh
```

## Training defaults

Both configs use 10 epochs, FP32 master/model parameters with BF16 autocast
compute, gradient checkpointing, a physical batch of 16 with accumulation 2
(effective batch 32), cosine scheduling, and a full fine-tuning learning rate
of `1e-5`. AdamW uses the same `betas=(0.9, 0.95)`, `eps=1e-8`, weight decay,
5% warm-up, and cosine-to-zero schedule as GenBridge's full stage. No
evaluation or checkpoint is written per epoch. This is
intentional because a full T5Gemma checkpoint is large.

Training refuses a non-empty output directory so an old `final_model` cannot
be mistaken for the new full-fine-tune result. For an intentional rerun, set
`OVERWRITE_OUTPUT_DIR=true` in the shell (or pass
`--overwrite-output-dir` to `train_full.py`). A `RUNNING` marker also prevents
evaluation of a crashed or incomplete run.

At startup, training prints the trainable/total parameter count and aborts
unless the trainable ratio is exactly 100%. If B200 memory is insufficient at
3072 source tokens, reduce physical batch size and increase accumulation by the
same factor to preserve effective batch 32.

## Outputs

Only the final full model is saved:

```text
runs/t5gemma2_1b_1b_full_wikilingua/final_model/
runs/lrsum_t5gemma2_1b_1b_full/final_model/
```

Each `final_model/` contains full safetensor weights, tokenizer files,
`training_config.yaml`, and `checkpoint_manifest.json`. Evaluation loads this
folder directly and writes `metrics.json`, `predictions.jsonl`, and
`eval_run_info.json` under the configured evaluation directory.

Manual commands:

```bash
python3 T5Gemma/scripts/train_full.py \
  --config T5Gemma/configs/wikilingua_full_3072.yaml

python3 T5Gemma/scripts/evaluate_full_test.py \
  --config T5Gemma/configs/wikilingua_full_3072.yaml \
  --checkpoint runs/t5gemma2_1b_1b_full_wikilingua/final_model \
  --test_file T5Gemma/data/processed/test.jsonl \
  --output_dir T5Gemma/eval_outputs/full_test
```
