#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Chuẩn bị môi trường cho T5Gemma ==="
if [[ -f "${ROOT_DIR}/src/T5Gemma/env.txt" ]]; then
  set -a; source "${ROOT_DIR}/src/T5Gemma/env.txt"; set +a
fi
"${PYTHON_BIN:-python3}" -m pip install -r "${ROOT_DIR}/src/T5Gemma/requirements.txt"
bash "${ROOT_DIR}/src/T5Gemma/run_xlsum_pipeline.sh"

echo "=== Chuẩn bị môi trường cho LLM2Seq ==="
if [[ -f "${ROOT_DIR}/src/llm2seq/env.txt" ]]; then
  set -a; source "${ROOT_DIR}/src/llm2seq/env.txt"; set +a
fi
"${PYTHON_BIN:-python3}" -m pip install -r "${ROOT_DIR}/src/llm2seq/requirements.txt"
bash "${ROOT_DIR}/src/llm2seq/scripts/run_xlsum_llama.sh"
bash "${ROOT_DIR}/src/llm2seq/scripts/run_xlsum_qwen.sh"
