#!/usr/bin/env bash
# ==============================================================================
# Script: Run ROUGE-1.5.5 scoring for seam_si_v2 (001, 002, 003)
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCORER_SCRIPT="src/evaluation/evaluate_rouge_vietnamese.py"
if [[ ! -f "$SCORER_SCRIPT" && -f "src/rouge155/evaluate_rouge_vietnamese.py" ]]; then
  SCORER_SCRIPT="src/rouge155/evaluate_rouge_vietnamese.py"
fi

for model in seam_si_v2_001 seam_si_v2_002 seam_si_v2_003; do
  echo "========================================================================"
  echo " Scoring ROUGE for: $model"
  echo "========================================================================"
  python3 "$SCORER_SCRIPT" --model-name "$model" --allow-missing
done
