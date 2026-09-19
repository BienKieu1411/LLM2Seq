# AFMR Adaptive Top-Down Keys

This folder is an isolated bridge experiment derived from `eviseq_new`. The encoder, decoder, cross-attention implementation, grounded-copy head, prompts, preprocessing, CE objective, optimizer, and generation settings are unchanged.

## Bridge design

```text
encoder final state H0
  ├─ base projection ──────────────────────────────────────── value memory + grounded copy
  └─ existing AFMR key path M
       └─ controller-weighted regional pooling
            └─ region position signal
                 └─ one global region mixer
                      └─ token-to-region top-down attention
                           └─ bounded low-rank correction ─── decoder key memory
```

The regional path:

1. Compacts content tokens only, excluding encoder instructions, special tokens, and padding.
2. Scores each compact token with a learned function conditioned on the existing document, decoder prompt, and output-budget controller.
3. Forms overlapping width-64, stride-48 regions with normalized learned weights. Prefix sums avoid a `[batch, region, width, hidden]` activation.
4. Adds normalized region-center/span features and mixes the compact region sequence with one 4-head self-attention layer at width 256.
5. Lets every content token attend to the mixed regions and maps the result through a rank-256 top-down correction.
6. Applies a document-conditioned gate (`0.05` initial, `0.30` maximum) and a smooth final cap of `0.30 × RMS(M)` per token.

The output factor starts with tiny nonzero RMS (`1e-3`), so CE reaches the pooling scorer, region mixer, token-to-region attention, and top-down projection on the first backward pass. No auxiliary loss, hard evidence selection, decoder-query router, contextual-value correction, distillation, or extra generated training data is used.

## Invariants

- The entire visible source remains in decoder memory.
- The adaptive correction changes decoder keys only.
- Base-projected `H0` remains the decoder value memory and grounded-copy memory.
- The original AFMR source prior remains unchanged.
- Rows with no content tokens remain finite and receive exactly zero correction.
- Non-content tokens receive exactly zero top-down correction.
- `bridge_mode: direct_projection` bypasses every bridge component and remains the w/o-bridge control.
- Main configs use FP32 parameters and updates, BF16 compute, greedy decoding, and `salience_loss_weight: 0`.

## Main configuration

The PubMed recipe is [`configs/afmr_pubmed.yaml`](configs/afmr_pubmed.yaml). Adaptive settings live under `architecture.adaptive_topdown` in [`configs/afmr_base.yaml`](configs/afmr_base.yaml):

```yaml
architecture:
  name: afmr_adaptive_topdown
  bridge_mode: afmr
  contextual_value:
    enabled: false
  adaptive_topdown:
    enabled: true
    dim: 256
    num_heads: 4
    mixer_layers: 1
    region_width: 64
    region_stride: 48
    topdown_rank: 256
    gate_init: 0.05
    gate_max: 0.30
    max_relative_rms: 0.30
    output_init_rms: 0.001
```

Checkpoints store the full nested bridge specification and reject incompatible architecture settings.

## PubMed training and evaluation

The queue uses local model paths and writes to a candidate-specific run root. `AFMR_ENCODERS=pplx` runs only the paper encoder.

```bash
cd /absolute/path/to/LLM2Seq/src/afmr_adaptive_topdown

PYTHON=python3 \
CUDA_VISIBLE_DEVICES=0,1 \
AFMR_ENCODERS=pplx \
PPLX_ENCODER=/absolute/path/to/pplx-embed-v1-0.6b \
DECODER_MODEL=/absolute/path/to/Qwen3-0.6B \
bash scripts/run_pubmed_pair.sh
```

Run the direct-projection control with a fresh output directory:

```bash
PYTHON=python3 \
CUDA_VISIBLE_DEVICES=0,1 \
AFMR_ENCODERS=pplx \
AFMR_BRIDGE_MODE=direct_projection \
AFMR_OUTPUT_DIR=/absolute/path/to/runs/adaptive_topdown_direct_control \
PPLX_ENCODER=/absolute/path/to/pplx-embed-v1-0.6b \
DECODER_MODEL=/absolute/path/to/Qwen3-0.6B \
bash scripts/run_pubmed_pair.sh
```

The runner trains, evaluates `last.pt`, and writes `last_test_predictions.jsonl`. It refuses to overwrite an existing checkpoint unless `OVERWRITE_OUTPUT_DIR=true` is explicitly set.

## Offline verification

No model download is required for the focused tests or smoke run:

```bash
cd /absolute/path/to/LLM2Seq/src/afmr_adaptive_topdown
PYTHONPATH=. /Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q tests/test_adaptive_topdown.py
PYTHON=/Users/kieugiangbien/bienkieu_env/bin/python bash scripts/run_afmr.sh smoke
```

The focused tests cover content-only pooling, prefix/padding invariance, finite all-invalid rows, the relative-RMS bound, first-backward gradients and parameter updates, direct-control isolation, the `H0` value anchor, nested config validation, and an end-to-end tiny CE backward pass.

Passing offline tests establishes graph and gradient correctness only. Whether this bridge improves ROUGE over `eviseq_new` and direct projection must be determined by matched training runs.
