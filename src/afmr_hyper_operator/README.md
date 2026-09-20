# AFMR Hyper Operator

This folder is a self-contained bridge experiment copied from `eviseq_new`.
It keeps the direct projection, copied decoder cross-attention and grounded-copy
anchor, while replacing the old AFMR depth/feature/focus path with a
document-conditioned low-rank K/V operator in every decoder layer.

For decoder layer `l`, the controller predicts mixture weights over `E`
low-rank experts. The selected expert residuals adapt the copied key and value
projections. Each layer owns independent experts. The value memory used by
grounded copy remains the projected final encoder state and is never replaced
by the adapted K/V tensors.

The expert up-projections start at zero, while the document router starts
with small nonzero weights so its conditioning path opens after the first
expert update. With the same seed, the full graph exactly matches
`direct_projection` before training. A smooth
relative-RMS cap bounds each adapted projection to 10% of its copied base
projection. The model trains with token-level cross-entropy only; salience
supervision and contextual-value branches are disabled.

## Default architecture

```yaml
architecture:
  name: afmr_hyper_operator
  hyper_operator:
    num_experts: 4
    rank: 16
    gate_init: 0.05
    gate_max: 0.20
    max_relative_rms: 0.10
    modulate_query: false
```

`modulate_query: false` gives the source bridge control of evidence retrieval
through K/V while preserving the decoder query geometry. The architecture adds
separate parameters per decoder layer, but it does not add source tokens or
increase the source context length.

## PubMed

Full architecture on one GPU:

```bash
cd src/afmr_hyper_operator
CUDA_VISIBLE_DEVICES=0 AFMR_ENCODERS=pplx \
PPLX_ENCODER=/local/path/to/pplx-embed-v1-0.6b \
DECODER_MODEL=/local/path/to/Qwen3-0.6B \
bash scripts/run_pubmed_pair.sh
```

Controlled w/o-bridge run, retaining direct projection and grounded copy:

```bash
cd src/afmr_hyper_operator
CUDA_VISIBLE_DEVICES=0 AFMR_ENCODERS=pplx \
AFMR_BRIDGE_MODE=direct_projection \
PPLX_ENCODER=/local/path/to/pplx-embed-v1-0.6b \
DECODER_MODEL=/local/path/to/Qwen3-0.6B \
bash scripts/run_pubmed_pair.sh
```

Use `CUDA_VISIBLE_DEVICES=0,1` for DDP. Set `AFMR_OUTPUT_DIR` to a fresh path
for every run. The runner evaluates `last.pt`; it does not save or select a
best checkpoint. Greedy generation uses `temperature: 0`, `top_k: 0`, and
`top_p: 1`.

## Offline verification

The architecture tests use the built-in tiny random backbones and never
download a model:

```bash
cd src/afmr_hyper_operator
/Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q tests/test_hyper_operator.py
```

The tests cover exact initialization parity with `direct_projection`, staged
gradient flow through the zero-output experts and controller, the RMS bound,
cached/uncached parity, and checkpoint architecture metadata.
