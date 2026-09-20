#!/usr/bin/env bash
set -Eeuo pipefail

CUDA_VISIBLE_DEVICES=0 \
AFMR_ENCODERS=pplx \
AFMR_BRIDGE_MODE=side_memory \
AFMR_GROUNDED_COPY=true \
AFMR_OUTPUT_DIR="$PWD/runs/full_side_memory_$(date +%Y%m%d_%H%M%S)" \
bash src/afmr_side_memory/scripts/run_pubmed_pair.sh

CUDA_VISIBLE_DEVICES=0 \
AFMR_ENCODERS=pplx \
AFMR_ARCHITECTURE=afmr_exact_routing \
AFMR_BRIDGE_MODE=afmr \
AFMR_GROUNDED_COPY=true \
AFMR_CONTEXTUAL_VALUE=false \
AFMR_SALIENCE_WEIGHT=0 \
AFMR_OUTPUT_DIR="$PWD/runs/full_exact_routing_$(date +%Y%m%d_%H%M%S)" \
bash src/afmr_exact_routing/scripts/run_pubmed_pair.sh
