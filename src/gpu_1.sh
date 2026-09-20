#!/usr/bin/env bash
set -Eeuo pipefail

CUDA_VISIBLE_DEVICES=1 \
AFMR_ENCODERS=pplx \
AFMR_ARCHITECTURE=afmr_hyper_operator \
AFMR_BRIDGE_MODE=afmr \
AFMR_GROUNDED_COPY=true \
AFMR_CONTEXTUAL_VALUE=false \
AFMR_SALIENCE_WEIGHT=0 \
AFMR_OUTPUT_DIR="$PWD/runs/full_hyper_operator_$(date +%Y%m%d_%H%M%S)" \
bash src/afmr_hyper_operator/scripts/run_pubmed_pair.sh

CUDA_VISIBLE_DEVICES=1 \
AFMR_ENCODERS=pplx \
AFMR_ARCHITECTURE=layerwise_coupled_depth_kv \
AFMR_BRIDGE_MODE=depth_kv \
AFMR_GROUNDED_COPY=true \
AFMR_OUTPUT_DIR="$PWD/runs/full_depth_kv_$(date +%Y%m%d_%H%M%S)" \
bash src/afmr_depth_kv/scripts/run_pubmed_pair.sh
