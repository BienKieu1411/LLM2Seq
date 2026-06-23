#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/load_env.sh"

required_files=(
  "${T5GEMMA_ROOT}/README.md"
  "${T5GEMMA_ROOT}/requirements.txt"
  "${T5GEMMA_ROOT}/configs/lora_l40_512.yaml"
  "${T5GEMMA_ROOT}/scripts/load_env.sh"
  "${T5GEMMA_ROOT}/scripts/prepare_wikilingua_json.py"
  "${T5GEMMA_ROOT}/scripts/train_lora.py"
  "${T5GEMMA_ROOT}/scripts/evaluate_full_test.py"
  "${T5GEMMA_ROOT}/scripts/train.sh"
  "${T5GEMMA_ROOT}/scripts/evaluate.sh"
  "${T5GEMMA_ROOT}/run_pipeline.sh"
  "${T5GEMMA_ROOT}/install_deps.sh"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Missing required file: ${file}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" -m py_compile \
  "${T5GEMMA_ROOT}/scripts/prepare_wikilingua_json.py" \
  "${T5GEMMA_ROOT}/scripts/train_lora.py" \
  "${T5GEMMA_ROOT}/scripts/evaluate_full_test.py" \
  "${T5GEMMA_ROOT}/scripts/push_to_hf.py"

bash -n "${T5GEMMA_ROOT}/scripts/load_env.sh"
bash -n "${T5GEMMA_ROOT}/scripts/train.sh"
bash -n "${T5GEMMA_ROOT}/scripts/evaluate.sh"
bash -n "${T5GEMMA_ROOT}/install_deps.sh"
bash -n "${T5GEMMA_ROOT}/run_pipeline.sh"

echo "T5Gemma folder smoke check OK."

