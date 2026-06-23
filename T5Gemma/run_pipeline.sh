#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/load_env.sh"
cd "${PROJECT_ROOT}"

echo "=== Prepare WikiLingua ==="
"${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/prepare_wikilingua_json.py" \
  --input_dir "${WIKI_DIR}" \
  --output_dir "${DATA_DIR}"

echo "=== Train T5Gemma LoRA ==="
bash "${T5GEMMA_ROOT}/scripts/train.sh"

if [[ "${RUN_EVAL,,}" == "true" || "${RUN_EVAL}" == "1" || "${RUN_EVAL,,}" == "yes" ]]; then
  echo "=== Evaluate Full Test ==="
  bash "${T5GEMMA_ROOT}/scripts/evaluate.sh"
fi

