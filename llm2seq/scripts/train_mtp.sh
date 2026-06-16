#!/bin/bash
# Train LLM2Seq with MTP only (resume from baseline)
set -euo pipefail

echo "=== LLM2Seq MTP-Only Training ==="
echo "MTP: ON (parallel) | Distillation: OFF"
echo "================================="

python -m llm2seq.src.training.trainer \
    --config llm2seq/configs/mtp_only.yaml

echo "Done! MTP-only training complete."
