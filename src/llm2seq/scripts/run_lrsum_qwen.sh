#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_VARIANT=qwen \
LRSUM_PHASE1_CONFIG="${LRSUM_PHASE1_CONFIG:-llm2seq/configs/lrsum_qwen_phase1.yaml}" \
LRSUM_PHASE2_CONFIG="${LRSUM_PHASE2_CONFIG:-llm2seq/configs/lrsum_qwen_phase2.yaml}" \
LRSUM_PHASE1_DIR="${LRSUM_PHASE1_DIR:-runs/lrsum_qwen_phase1_warmup}" \
LRSUM_PHASE2_DIR="${LRSUM_PHASE2_DIR:-runs/lrsum_qwen_phase2_lora_encoder}" \
bash "${SCRIPT_DIR}/run_lrsum_pipeline.sh"
