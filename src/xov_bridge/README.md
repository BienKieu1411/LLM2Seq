# Cross-Tokenizer Ordered Value Bridge (XOV)

Independent experiment in `src/xov_bridge`, package `xov`. It does not modify or
import `eviseq_new`. The pipeline derives from the historical XOV experiment;
data and prompt handling follow the current baseline. Pretrained models and
tokenizers load locally. `__tiny__` creates random test models without downloads.

## Bridge

```
encoder H0 -> direct projection X -> cross-attention keys and copy context
source decoder-token embeddings -> RMSNorm -> down -> gap-masked depthwise conv
    -> SiLU -> overlap pooling -> up -> bounded residual -> X + residual (values)
```

Alignment uses only visible source spans and retains original token ordinals.
Convolution cannot join tokens separated by removed tokens. Prefix-crossing
encoder spans are clipped consistently; unaligned rows retain X. Missing
alignment is an error even with copy disabled. Copy context stays on X, but its
probabilities can still change through decoder hidden states and shared training.

## Active initialization

In `configs/xov_base.yaml`:

```yaml
architecture:
  output_init_gain: 1.0
  lexical_rank: 256
  phrase_kernel: 3
  value_gate_init: 0.10
  value_gate_max: 0.20
  residual_reference_rms: 1.0
```

The up-projection uses nonzero orthogonal initialization. All lexical layers can
receive gradients on the first backward pass, unlike zero-up initialization.
The gate starts at 0.10 rather than 0.05. Its maximum bounds the **pre-decoder**
relative residual RMS; 0.10 does not mean an actual 10% residual. A synthetic
1024-wide/256-rank fixture measured a median of 1.12%. Real model signal depends
on embeddings, alignment, and normalization. The gate remains trainable; no
minimum contribution or artificial loss forces the branch to dominate.

This intentionally overrides the research design's original zero-up choice at
the user's request. It is an inductive bias, not evidence of a ROUGE gain.

## Train and evaluate

Edit local encoder/decoder paths, data paths, batch sizes, and epoch counts in
`configs/xov_pubmed.yaml` (inherits `xov_base.yaml`). `training.batch_size` is per
GPU. Prepared canonical rows use `id`, `text`, `summary`; field names can be
configured. No per-row system prompt is consumed. Encoder prefix and decoder
prompt/prefix are configured separately; decoder instructions do not reduce the
encoder source budget.

From this folder, train and evaluate on two GPUs in one command:

```bash
GPU_IDS=0,1 EVAL_BATCH_SIZE=64 bash scripts/train_eval_2gpu.sh
# Optional config path is resolved relative to your current directory.
# SPLIT=validation selects validation instead of test.
```

Individual commands:

```bash
# One GPU
CUDA_VISIBLE_DEVICES=0 python3 run_xov.py train configs/xov_pubmed.yaml

# Two GPUs, DDP
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run --standalone \
  --nproc_per_node=2 run_xov.py train configs/xov_pubmed.yaml

# Evaluate last checkpoint; use the resolved config saved by training.
CUDA_VISIBLE_DEVICES=0 python3 run_xov.py evaluate \
  runs/xov/pubmed_value_anchor_copy/resolved_config.yaml \
  runs/xov/pubmed_value_anchor_copy/last.pt \
  runs/xov/pubmed_value_anchor_copy/test_predictions.jsonl --split test
```

`bash scripts/run_xov.sh` forwards the same CLI arguments. The optional
`run_pubmed_pair.sh` runs the PubMed encoder comparison sequentially and invokes
external ROUGE-1.5.5. Built-in Python ROUGE is only a diagnostic; use the same
ROUGE-1.5.5 protocol as the baseline for reported comparisons.

Generation supports greedy decoding (`temperature: 0`, `top_k: 0`, `top_p: 1`)
and sampling with `do_sample: true` and positive temperature. Evaluation supports
`--shard-rank` / `--num-shards`; each shard must write to a distinct output file.

Checkpoint contracts include operator semantics and hashes of local model config
and tokenizer assets. Same-shaped historical XOV checkpoints are rejected.
Resume restores optimizer/scheduler and per-rank RNG with the same world size;
this does not guarantee bitwise GPU reproducibility. To compare the bridge,
train a separate `bridge_mode: direct_projection` control with matched settings.

## Offline checks

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m pytest tests -q
python3 scripts/smoke_test.py
python3 -m ruff check .
python3 -m ruff format --check .
```

Tests use tiny Qwen models, including CE weight updates, copy routing, active
initialization, alignment gaps/boundaries, empty rows, BF16 CPU autocast, cache
compaction, and checkpoint roundtrip. No pretrained task scores follow from them.
