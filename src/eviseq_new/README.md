# EviSeq AFMR

AFMR (Adaptive Full-Memory Residual) is a single-memory encoder–decoder architecture for general text summarization:

```text
pretrained encoder → bounded AFMR bridge → copied cross-attention Qwen decoder → greedy generation
```

The bridge preserves every valid source token. A document/prompt/requested-budget controller conditions a low-rank depth residual in encoder space, a low-rank encoder-to-decoder feature residual, and one multi-scale source prior. Depth weights have shape `[batch, source_tokens, depth_taps]`: a shared learned scorer reads each normalized candidate token representation, plus a document-conditioned depth preference. Softmax is over depth, not tokens; selection runs once after the encoder. Residual output factors and the depth/focus scorers start at zero; gates and prior strength are bounded, but learned residual vector norms are not mathematically bounded. The prior is additive `[batch, source_tokens]` and is consumed by every decoder cross-attention layer. There are no banks, hard top-k pruning, per-layer routers, rerankers, beam search, KD, or test-time evidence labels.

Focus regions are overlapping **token windows**, not selected sentences. Decoder memory contains the entire source. The only objective is token-level cross-entropy (CE): depth, feature and focus routes all learn through it. There is no positive-sentence mining, allocation loss, or contrastive head. External evidence labels are ignored.

## Offline smoke test

The smoke test is deterministic, uses only tiny randomly initialized modules, and never calls Hugging Face. It checks tensor contracts, special-token masking, zero-initialized residuals, CE gradients, optimizer updates, and checkpoint round-trip:

```bash
cd /absolute/path/to/LLM2Seq
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  src/eviseq_new/scripts/run_afmr.sh smoke
```

Expected output is one JSON object with `"status": "ok"`. Local pytest is also model-free:

```bash
cd src/eviseq_new
PYTHONPATH=. /absolute/path/to/bienkieu_env/bin/pytest -q
```

To smoke the real checkpoints on an A100 without a long run, use a task config with local model/data paths and cap train, validation, and test:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON=python3 \
  bash scripts/smoke_a100.sh configs/afmr_pubmed.yaml
```

This exercises the actual tokenizer, encoder, AFMR bridge, copied
cross-attention, backward pass, checkpoint writer, and greedy KV-cache
generation. Offline mode is enforced: configured models must already be available locally. Set `AFMR_SMOKE_EVAL_EXAMPLES` and
`AFMR_SMOKE_EVAL_BATCH_SIZE` to control the bounded evaluation; set
`AFMR_SMOKE_DEVICE=cpu` only when exercising the script locally without CUDA.
The script creates an isolated `runs/smoke/run_*` directory and never overwrites the task's full-training directory. It runs one warm-up and one full epoch with batch 2 × accumulation 2; task architecture and source/target limits are preserved. Set `AFMR_SMOKE_TRAIN_EXAMPLES` and `AFMR_SMOKE_VALIDATION_EXAMPLES` to change the default 100/20 caps.

## Preparing data

Each input record contains the configured source and target fields (strings or lists of strings). Training can read raw JSONL directly. For the same three-split preparation used by `eviseq_v2`, run `prepare-dataset`: PubMed/ArXiv expect `train.label.jsonl`, `val.label.jsonl`, and `test.label.jsonl`; CNNDM/WikiLingua accept the usual `train`, `val`/`validation`, and `test` JSONL/JSON/TXT names. The command copies raw files, writes canonical `id/text/summary` JSONL, emits `preparation_report.json`, and rejects duplicate IDs or exact source text across splits. Evidence labels are not consumed by AFMR.

```bash
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  bash scripts/prepare_afmr.sh --dataset pubmed \
  --input-dir /data/pubmed --output-dir datasets/pubmed \
  --raw-copy-dir data/raw/pubmed
```

The same command handles ArXiv, CNN/DailyMail, and WikiLingua by changing
`--dataset` and `--input-dir`:

```bash
bash scripts/prepare_afmr.sh --dataset arxiv --input-dir /data/arxiv --output-dir datasets/arxiv
bash scripts/prepare_afmr.sh --dataset cnndm --input-dir /data/cnndm --output-dir datasets/cnndm
bash scripts/prepare_afmr.sh --dataset wikilingua --input-dir /data/wikilingua --output-dir datasets/wikilingua
```

## A100 training/evaluation

Start from `configs/afmr_base.yaml`, copy it to a task recipe, and set only model locations, data files, lengths, batch resources, and output directory. The generic base uses one warm-up epoch and four full-finetuning epochs; benchmark recipes override this to match the corresponding T5Gemma total (PubMed 1+3, CNNDM/WikiLingua 1+5). Decoding is greedy (`num_beams: 1`, `do_sample: false`), and training is CE-only.

```bash
cd src/eviseq_new
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_afmr.sh train configs/afmr_pubmed.yaml

PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_afmr.sh evaluate \
  configs/afmr_pubmed.yaml runs/afmr/pubmed/last.pt \
  runs/afmr/pubmed/test_predictions.jsonl --split test
```

The runtime loads models only for `train` or `evaluate`; importing AFMR and running tests does not download anything. Checkpoints are structurally guarded: changing batch size, generation batch size, data paths, or model folder location is allowed, while changing AFMR ranks, windows, depth taps, or cross-attention layout is rejected.

For a controlled T5Gemma comparison, each task recipe copies the exact T5Gemma
source instruction and uses an empty decoder prompt. The collator then falls
back to the decoder BOS/EOS start token, matching the native seq2seq decoder
contract instead of giving AFMR an extra textual instruction. Greedy decoding
also applies the same repetition penalty, no-repeat n-gram constraint, minimum
length, and maximum length as the corresponding T5Gemma recipe. A richer
decoder prompt is a separate ablation and requires a separately fine-tuned
checkpoint.

The token-wise graph (`afmr_token_depth_lowrank_v3`) is intentionally incompatible with earlier document-wise AFMR checkpoints. Do not use `strict=False` to force-load those checkpoints. Train this graph from pretrained backbones in a separate output directory. New token-wise checkpoints resume normally.

Training uses FP32 bridge arithmetic, BF16 or FP32 backbones, non-reentrant backbone checkpointing, token-weighted gradient accumulation (including a partial final window), and a per-stage linear LR decay. Optimizer moments are carried from warm-up to full fine-tuning. LM-head CE is computed in checkpointed token chunks instead of retaining full `[B,T,V]` logits. Encoder KV caching is disabled; only the requested depth taps are captured. The runner currently supports one GPU/process and rejects multi-process launches.

Training prints Trainer-style progress lines with stage, global epoch, optimizer step, token-weighted CE, gradient norm, learning rates, elapsed time, examples and token throughput. The same step/epoch records are appended to `training_metrics.jsonl`; epoch summaries report train/validation CE. Generation writes every completed sample batch immediately to JSONL, prints progress/ETA, resumes from a contiguous prefix, retries CUDA OOM by halving the active batch, and stores final metrics in `<predictions>.metrics.json`.

Checkpoints are `epoch_001.pt`, ..., `last.pt`; `best.pt` is optional and off in the main recipe. Final evaluation uses `last.pt`. `generation.batch_size` controls decoding; `--batch-size` overrides it explicitly. An existing prediction file resumes only if its IDs and references match a contiguous prefix of the active split. Each new batch is flushed to JSONL with progress/ETA. Use a different output filename when comparing another checkpoint. Evaluation retries CUDA OOM by halving the active batch; if a single example still fails, it raises the original error.

All local verification uses tiny random backbones with network disabled, including BF16 attention backward, CE-only focus gradients, cache parity, accumulation and warm-up resume equivalence. A real PPLX/Qwen GPU smoke and throughput profile remain necessary; local tests do not establish B200 kernel efficiency or ROUGE improvement.

For bundled `configs/*.yaml`, relative data/output paths are rooted at `eviseq_new/`, independent of the calling directory or whether files already exist. An external YAML resolves relative paths against its own directory. At encoder/decoder hidden size 1024, the main AFMR bridge contains 1,981,446 parameters; this excludes both pretrained backbones and the copied decoder cross-attention. Token-wise depth selection adds 1,024 parameters over the previous document-wise graph and does not stack a second full-depth memory tensor.

## Package layout

```text
eviseq_new/
├── eviseq_afmr/       public AFMR namespace
│   ├── data/           schema, split preparation, dataset, collator
│   ├── modeling/      encoder, controller, bridge, decoder, model
│   ├── training/      optimizer, checkpoint, engine
│   └── evaluation/    greedy generation, metrics
├── configs/            AFMR base and task recipes
├── scripts/run_afmr.sh source-tree entry point
├── scripts/smoke_a100.sh bounded real-model smoke run
└── tests/              model-free contract and gradient tests
```
