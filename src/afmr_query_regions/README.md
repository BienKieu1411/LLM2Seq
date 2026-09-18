# AFMR query-region bridge (experimental)

This folder is an isolated copy of the PubMed AFMR pipeline. It leaves
`src/eviseq_new` available as the static-bridge control. The new route reads
source regions using the **current decoder state**: it pools region keys from
the AFMR memory `M`, pools region values from the original projected encoder
states `H0`, and adds a bounded residual to each decoder layer's query before
the existing full-token cross-attention. Grounded copy, the `H0` value anchor,
and the original token-level source path remain in place. The added route is
zero at initialization; its output projection learns on the first update and
its query/key/value projections receive gradient after that update.

This is a testable architecture candidate, **not a measured ROUGE gain**. The
separate [research delta](../../doc/research/bridge_contribution_2026_09_18/diffs/2026-09-18_query_region_delta.md)
records its rationale and counterarguments. In particular, these components
have precedents in the literature; a novelty claim requires a precise method
description and matched ablations.

## Run three matched PubMed arms

Use local encoder/decoder paths already configured in `scripts/run_pubmed_pair.sh`
or override `PPLX_ENCODER`, `DECODER_MODEL`, and `PUBMED_SOURCE_DIR`. The runner
prepares the dataset in this folder if needed and then trains/evaluates
`last.pt`. Run one arm at a time; keep training settings identical and use
distinct output directories. These commands use one visible GPU. For a
two-process DDP run, launch `run_afmr.py train` with `torchrun` and a
materialized config instead of assuming that `CUDA_VISIBLE_DEVICES=0,1`
automatically starts two processes.

```bash
cd src/afmr_query_regions
export CUDA_VISIBLE_DEVICES=0 AFMR_ENCODERS=pplx AFMR_SALIENCE_WEIGHT=0

# A: w/o bridge; encoder, decoder and grounded copy remain present.
AFMR_BRIDGE_MODE=direct_projection AFMR_REGION_QUERY=false \
  AFMR_OUTPUT_DIR="$PWD/runs/A_direct" bash scripts/run_pubmed_pair.sh

# B: current static AFMR bridge, cross-attention and copy.
AFMR_BRIDGE_MODE=afmr AFMR_REGION_QUERY=false \
  AFMR_OUTPUT_DIR="$PWD/runs/B_static" bash scripts/run_pubmed_pair.sh

# C: same B plus decoder-conditioned regional read.
AFMR_BRIDGE_MODE=afmr AFMR_REGION_QUERY=true \
  AFMR_OUTPUT_DIR="$PWD/runs/C_query_regions" bash scripts/run_pubmed_pair.sh
```

`C−B` isolates the new regional read. `B−A` tests the static bridge. `C−A`
tests the complete bridge. All three recipes use cross-entropy only, with the
same grounded-copy route and greedy test decoding. Compare resolved configs,
checkpoint epoch, update count, preprocessing fingerprints, seed and ROUGE
version before interpreting scores. Select configuration on validation; use
test once after it is fixed. This copied pipeline contains no PubMed data or
pretrained checkpoint.

## Local verification

From the repository root:

```bash
PYTHONPATH=src/afmr_query_regions \
  /Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q \
  src/afmr_query_regions/tests
```

The tiny fixtures use `__tiny__` models and do not download full weights.
They check exact initialization parity against static AFMR, CE gradient/weight
updates, FP32 gradients under BF16 autocast, bounded query perturbation,
all-invalid source masks, and cached generation parity. GPU throughput, VRAM,
DDP behavior and PubMed ROUGE still require the server run.
