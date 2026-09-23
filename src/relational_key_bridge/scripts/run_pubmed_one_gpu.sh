#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CONFIG="$ROOT/configs/relational_key_pubmed.yaml"
RUN_DIR="${RUN_DIR:-$ROOT/runs/pubmed_relational_key_$(date +%Y%m%d_%H%M%S)}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
ROUGE155_SCRIPT="${ROUGE155_SCRIPT:-$ROOT/../rouge155/evaluate_rouge.py}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Relational-key PubMed; GPU=$CUDA_VISIBLE_DEVICES; batch/GPU=84; accumulation=1 ==="
echo "=== Run: $RUN_DIR ==="
"$PYTHON" "$ROOT/run_relational_key.py" train "$CONFIG" --output-dir "$RUN_DIR"
"$PYTHON" "$ROOT/run_relational_key.py" evaluate \
  "$RUN_DIR/resolved_config.yaml" "$RUN_DIR/last.pt" "$RUN_DIR/test_predictions.jsonl" \
  --split test --batch-size "$EVAL_BATCH_SIZE"
if [[ -f "$ROUGE155_SCRIPT" ]]; then
  "$PYTHON" "$ROUGE155_SCRIPT" "$RUN_DIR/test_predictions.jsonl"
fi
