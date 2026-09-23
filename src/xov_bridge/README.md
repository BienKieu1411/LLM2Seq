# Cross-Tokenizer Ordered Key-Value Bridge (XOV)

Independent experiment in `src/xov_bridge`, package `xov`. It does not modify or
import `eviseq_new`. The pipeline derives from the historical XOV experiment;
data and prompt handling follow the current baseline. Pretrained models and
tokenizers load locally. `__tiny__` creates random test models without downloads.

## Bridge

```
encoder H0 -> direct projection X -> copy context and key/value anchors
source decoder-token embeddings -> RMSNorm -> down -> gap-masked depthwise conv
    -> SiLU -> overlap pooling -> up -> bounded residual R
cross-attention keys   = X + g_key R
cross-attention values = X + g_value R
grounded-copy memory   = X
```

Alignment uses only visible source spans and retains original token ordinals.
Convolution cannot join tokens separated by removed tokens. Prefix-crossing
encoder spans are clipped consistently; unaligned rows retain X. Missing
alignment is an error even with copy disabled. Both gates are independently
trained and nonzero at initialization; the key gate starts at 0.12 and is
bounded by 0.20, while the value gate keeps its configured initialization and
cap. The existing `key_memory: direct_projection` setting denotes the anchor
X; the final cross-attention key memory includes the bounded lexical residual.
Copy context stays exactly on X, but its probabilities can still change through
decoder hidden states and shared training.

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
The value gate starts at 0.10 and the key gate at 0.12. Their maxima bound the
**pre-decoder** relative residual RMS; a gate of 0.10 does not imply an actual
10% residual. Real model signal depends on embeddings, alignment, and
normalization. Both gates remain trainable; no
minimum contribution or artificial loss forces the branch to dominate.
A local synthetic check with 1024-dimensional states, rank 256, 128 aligned
positions, and fixed random queries found mean post-normalization/key-projection
attention total-variation shifts of 0.0044 at key-gate init 0.12 versus 0.0018
at 0.05. This confirms an active retrieval route in that fixture; it does not
estimate the effect with pretrained weights or predict ROUGE.

This intentionally overrides the research design's original zero-up choice at
the user's request. It is an inductive bias, not evidence of a ROUGE gain.

The PubMed experiment uses `architecture.value_gate_mode: source_lexical`.
The value gate reads each position's encoder state and aligned lexical features,
so it can reduce or increase the lexical residual locally. Its added weights
start at zero, making its initial value exactly the previous global value gate.
The independent key gate makes aligned lexical evidence visible to decoder
retrieval after memory normalization and key projection. This is the specific
mechanism the earlier value-only XOV lacked. `global` remains available as a
controlled value-gate comparison. Old value-only checkpoints cannot be loaded
into this graph.
Input-dependent gates have precedent in [Gated Multimodal Units](https://arxiv.org/abs/1702.01992);
that paper does not test this summarization bridge or predict a ROUGE gain.

## Train and evaluate

The PubMed config uses local model paths from the project server and inherits
`xov_base.yaml`. Its current settings are zero interface warmup epochs, three
full epochs, batch 84 per GPU, accumulation 1, and gradient clip 3.0.
`training.batch_size` is per GPU; one and two GPU runs therefore use different
global batch sizes. Prepared canonical rows use `id`, `text`, `summary`; field names can be
configured. No per-row system prompt is consumed. Encoder prefix and decoder
prompt/prefix are configured separately; decoder instructions do not reduce the
encoder source budget.

From this folder, train and evaluate on two GPUs in one command:

```bash
GPU_IDS=0,1 SPLIT=validation EVAL_BATCH_SIZE=64 bash scripts/train_eval_2gpu.sh
# Optional config path is resolved relative to your current directory.
# Run test only after choosing the architecture on validation.
```

Individual commands:

```bash
# One GPU with the YAML settings unchanged
CUDA_VISIBLE_DEVICES=0 python3 run_xov.py train configs/xov_pubmed.yaml

# Two GPUs, DDP
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run --standalone \
  --nproc_per_node=2 run_xov.py train configs/xov_pubmed.yaml

# Evaluate last checkpoint; use the resolved config saved by training.
CUDA_VISIBLE_DEVICES=0 python3 run_xov.py evaluate \
  runs/xov/pubmed_source_lexical_matched/resolved_config.yaml \
  runs/xov/pubmed_source_lexical_matched/last.pt \
  runs/xov/pubmed_source_lexical_matched/test_predictions.jsonl --split test
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
train a separate `bridge_mode: direct_projection` control with matched settings
when a new control is needed. The reported value-only XOV result,
49.534/22.028/45.815, predates the key route and cannot establish its benefit.
The key/value design is a testable retrieval-and-transfer hypothesis; it does
not guarantee a ROUGE improvement or establish novelty by itself.

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
