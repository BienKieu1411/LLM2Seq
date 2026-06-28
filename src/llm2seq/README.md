# LLM2Seq Training Package

This folder contains the main LLM2Seq implementation for Vietnamese abstractive summarization.

LLM2Seq reuses pretrained LLM-based source encoders inside an encoder-decoder summarizer. Encoder states are mapped into decoder cross-attention memory through a gated residual adapter. After the main summarizer is trained, verified Multi-Token Prediction (MTP) heads are trained for faster decoding.

## Folder Layout

```text
src/llm2seq/
  configs/              # WikiLingua and VLSP training configs
  scripts/              # preprocessing, training, evaluation, HF utilities
  src/
    data/               # dataset and collator code
    inference/          # autoregressive and verified MTP decoding
    models/             # encoder wrapper, adapter, decoder, MTP modules
    training/           # losses, trainer, schedulers
  requirements.txt
  run_pipeline.sh
  install_deps.sh
  smoke_check.sh
  env.example.txt
```

## Main WikiLingua Configs

```text
configs/wikilingua_qwen_phase1.yaml
configs/wikilingua_qwen_phase2.yaml
configs/wikilingua_qwen_phase3.yaml
configs/wikilingua_phase1.yaml
configs/wikilingua_phase2.yaml
configs/wikilingua_phase3.yaml
```

The Qwen configs use `Qwen/Qwen3-Embedding-0.6B` as the source encoder. The Llama-oriented configs use the LLM2Vec-based encoder setup.

## Setup

From this folder:

```bash
bash smoke_check.sh
bash install_deps.sh
cp env.example.txt env.txt
```

Edit `env.txt` for local paths, Hugging Face token, upload repository, and Python executable.

Then run:

```bash
bash run_pipeline.sh
```

The scripts set `PYTHONPATH` so imports resolve to this local package.

## Data

Expected WikiLingua files:

```text
datasets/wikilingua/train.json
datasets/wikilingua/val.json
datasets/wikilingua/test.json
```

Each record contains:

```json
{"src": ["source sentence"], "tgt": ["summary sentence"]}
```

The preprocessing script joins source sentences into one document and target sentences into one summary paragraph.

## Training Phases

Phase 1 trains the adapter, decoder, memory tokens, and LM head while keeping the encoder frozen.

Phase 2 adds LoRA adaptation to the source encoder and continues summarization training.

Phase 3 freezes the main summarizer and trains only the cascaded MTP heads with self-distillation.

## Evaluation

Full-test evaluation scripts are under:

```text
scripts/evaluate_full_test.py
scripts/compute_bertscore.py
scripts/compare_speed_metrics.py
```

The report uses ROUGE, BERTScore, length statistics, and runtime metrics generated from these scripts.

## Notes

- `env.txt`, local checkpoints, processed data, and prediction files are intentionally ignored by git.
- Use the root `README.md` for the demo app and report workflow.
- Use `deploy/README.md` for Docker and Kubernetes deployment.
