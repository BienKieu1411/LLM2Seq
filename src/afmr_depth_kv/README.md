# Layerwise Coupled-Depth K/V

This independent bridge experiment keeps the final encoder state as the
direct-projected source and grounded-copy anchor. Unlike one shared bridge
memory for all decoder layers, it builds a token-aligned memory for each
decoder layer. A document/prompt/budget-conditioned router mixes four encoder
depth taps. A low-rank residual then changes the layer's memory by at most a
configured fraction of the final-state anchor RMS. **The same adapted memory
supplies both K and V** in that layer; grounded copy continues to read the
unchanged final-state anchor.

The residual output factors start at zero, so full and `direct_projection`
produce identical source memories at initialization. CE first trains the
output factors, then opens gradients to the upstream depth router. No
contrastive, salience or candidate-generation objective is used.

The PubMed config uses batch 16 × accumulation 6 on one GPU, preserving an
effective batch of 96 while controlling the extra per-layer source-memory
activations. This variant is likely the most memory-intensive of the four;
measure actual B200 peak VRAM before increasing the microbatch.

## Run

```bash
cd src/afmr_depth_kv
CUDA_VISIBLE_DEVICES=0 AFMR_ENCODERS=pplx \
PPLX_ENCODER=/local/path/to/pplx-embed-v1-0.6b \
DECODER_MODEL=/local/path/to/Qwen3-0.6B \
bash scripts/run_pubmed_pair.sh
```

For the direct-projection + grounded-copy ablation, also set
`AFMR_BRIDGE_MODE=direct_projection` and a fresh `AFMR_OUTPUT_DIR`. Use
`CUDA_VISIBLE_DEVICES=0,1` for DDP. The runner evaluates `last.pt` and uses
greedy generation with temperature 0, top-k 0, top-p 1.

The folder has its own package, configs, scripts and tests. Offline tests use
tiny random backbones and download no model:

```bash
cd src/afmr_depth_kv
/Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q tests
```

These tests establish shape, gradient, optimizer, copy-anchor and cache
correctness. Whether the bridge improves ROUGE over the direct projection
requires a matched training and evaluation run.
