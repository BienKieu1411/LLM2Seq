# AFMR projected-query region bridge (experimental)

This folder is a separate architecture candidate. `src/eviseq_new` remains
the static bridge control and `src/afmr_query_regions` remains the original
hidden-space regional bridge control. No PubMed checkpoint or ROUGE result is
included here.

The regional bank pools overlapping content-token windows. In full AFMR,
region keys come from semantic memory `M` and region values from the projected
encoder anchor `H0`. In `direct_projection` mode, both come from projected
`H0`. Every selected decoder layer reads this bank with its current state.
The regional output is zero at initialization. After learning, its residual
is bounded **per token and per cross-attention head relative to the projected
Q**, then added before Q normalization and ordinary full-token cross-attention.
This addresses the prior version's weaker hidden-space bound. It does not
mathematically bound changes to final attention probabilities or ROUGE.

Grounded copy, the original token-level cross-attention and CE-only training
remain in place. The optional direct+region arm is a scientific control: it
tests whether the new regional read needs AFMR's static depth/feature/focus
path. This folder does not add multi-scale regions, position embeddings,
dynamic gates or a different training loss because those would confound the
projected-Q test.

## Matched PubMed arms

Run from this folder, using local paths in `scripts/run_pubmed_pair.sh` or
overriding `PPLX_ENCODER`, `DECODER_MODEL` and `PUBMED_SOURCE_DIR`.
Each arm needs a different output directory. The script trains and evaluates
`last.pt` on one visible GPU. Two visible GPUs alone do not launch DDP.
In direct-projection mode, `AFMR_REGION_QUERY` must be set explicitly so A
cannot be mistaken for D.
For parallel C and D runs, set `PROCESSED_DATA_DIR` to the same already
prepared PubMed directory; each run writes its own config, checkpoint and log.

```bash
cd src/afmr_query_qspace
export CUDA_VISIBLE_DEVICES=0 AFMR_ENCODERS=pplx AFMR_SALIENCE_WEIGHT=0

# A — projected H0 + ordinary cross-attention + grounded copy.
AFMR_BRIDGE_MODE=direct_projection AFMR_REGION_QUERY=false \
  AFMR_OUTPUT_DIR="$PWD/runs/A_direct" bash scripts/run_pubmed_pair.sh

# B — static AFMR bridge, no regional read.
AFMR_BRIDGE_MODE=afmr AFMR_REGION_QUERY=false \
  AFMR_OUTPUT_DIR="$PWD/runs/B_static" bash scripts/run_pubmed_pair.sh

# C — static AFMR plus projected-Q regional read.
AFMR_BRIDGE_MODE=afmr AFMR_REGION_QUERY=true \
  AFMR_OUTPUT_DIR="$PWD/runs/C_afmr_qspace" bash scripts/run_pubmed_pair.sh

# D — projected H0 plus projected-Q regional read, no static AFMR path.
AFMR_BRIDGE_MODE=direct_projection AFMR_REGION_QUERY=true \
  AFMR_OUTPUT_DIR="$PWD/runs/D_direct_qspace" bash scripts/run_pubmed_pair.sh
```

`C−B` isolates the new read on AFMR; `D−A` isolates it on direct projection;
`C−D` tests whether the static AFMR path adds value in the presence of the
new read. To isolate **projected-Q versus hidden-space** placement, also run
the original `src/afmr_query_regions` full configuration with the same
training and evaluation settings. Use identical local checkpoints, dataset,
prompt, lengths, seed, batch×accumulation, optimizer, update count, clip,
greedy decoding and ROUGE-1.5.5. Inspect every `resolved_config.yaml` before
interpreting scores. Choose settings on validation, then evaluate test once.

## Local verification

From the repository root, using the lightweight `__tiny__` fixtures:

```bash
PYTHONPATH=src/afmr_query_qspace \
  /Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q \
  src/afmr_query_qspace/tests
```

Tests require no full model download. They verify initial parity, per-head
projected-Q cap, CE gradients and optimizer updates, BF16/FP32 handling,
direct+region configuration and cached generation. Server-side B200 memory,
speed, DDP and ROUGE still need measurement.
