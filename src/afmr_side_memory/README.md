# Gated Side-Memory Bridge

This folder is a self-contained experimental architecture derived from the
stable `eviseq_new` training and evaluation pipeline. It keeps three paths
unchanged:

1. the final encoder token states are projected directly into the decoder;
2. every decoder layer reads those exact tokens with its original base
   cross-attention branch;
3. grounded copy is prepared only from those exact projected tokens.

The new branch resamples the final encoder states into 24 compact evidence
tokens. Every fourth decoder layer reads this side bank with an independent
K/V projection and an independent softmax. The two banks are never
concatenated. Side output is added through a scalar `tanh` gate initialized to
exactly zero and a smooth per-token relative-RMS cap of 0.10. Therefore, the
full graph exactly matches the direct-projection control at initialization,
while CE can first open the gate and then train the resampler and side
attention.

Training uses token cross entropy only. There is no salience, contrastive,
contextual-value, region, depth-mixture, or generated-candidate loss.

## Configuration

The architecture-specific defaults are in `configs/afmr_base.yaml`:

```yaml
architecture:
  name: afmr_side_memory
  bridge_mode: side_memory
  depth_taps: 1
  side_tokens: 24
  side_num_heads: 8
  side_resampler_dropout: 0.0

decoder:
  cross_attention_every: 1
  side_attention_every: 4
  side_relative_rms_cap: 0.10
  grounded_copy:
    enabled: true
```

Set `architecture.bridge_mode: direct_projection` for the controlled ablation.
The direct control retains the encoder, base projection, decoder
cross-attention and grounded-copy head; only side-memory creation/use is
disabled.

## PubMed

One GPU:

```bash
CUDA_VISIBLE_DEVICES=0 AFMR_ENCODERS=pplx \
  bash src/afmr_side_memory/scripts/run_pubmed_pair.sh
```

Two-GPU DDP training:

```bash
CUDA_VISIBLE_DEVICES=0,1 AFMR_ENCODERS=pplx \
  bash src/afmr_side_memory/scripts/run_pubmed_pair.sh
```

The runner infers one process per visible GPU for training. Evaluation remains
single-process and uses a separate side K/V cache during autoregressive
generation. It preserves `temperature`, `top_k`, and `top_p` from the resolved
YAML.

Controlled w/o-side-memory run:

```bash
CUDA_VISIBLE_DEVICES=0 AFMR_ENCODERS=pplx \
  AFMR_BRIDGE_MODE=direct_projection \
  AFMR_OUTPUT_DIR="$PWD/runs/side_memory_direct_projection" \
  bash src/afmr_side_memory/scripts/run_pubmed_pair.sh
```

Useful overrides are `AFMR_SIDE_TOKENS` (16--32),
`AFMR_SIDE_ATTENTION_EVERY`, `AFMR_SIDE_RMS_CAP`, `AFMR_GROUNDED_COPY`,
`PROCESSED_DATA_DIR`, and `AFMR_OUTPUT_DIR`.

## Offline checks

No test downloads a model:

```bash
/Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q src/afmr_side_memory/tests
/Users/kieugiangbien/bienkieu_env/bin/python src/afmr_side_memory/scripts/smoke_test_side_memory.py
```

The focused tests cover exact zero-gate parity, staged gradient flow, optimizer
updates, base/copy isolation, active-branch effect, and cached generation
parity.
