#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/src/llm2seq/scripts/load_env.sh"

if [[ "${SKIP_INSTALL_DEPS:-false}" != "true" && "${SKIP_INSTALL_DEPS:-false}" != "1" ]]; then
  echo "=== Install LLM2Seq dependencies ==="
  bash "${ROOT_DIR}/src/llm2seq/install_deps.sh"
fi

bash "${ROOT_DIR}/src/llm2seq/scripts/run_lrsum_llama.sh"
bash "${ROOT_DIR}/src/llm2seq/scripts/run_lrsum_qwen.sh"

if [[ "${SKIP_INSTALL_DEPS:-false}" != "true" && "${SKIP_INSTALL_DEPS:-false}" != "1" ]]; then
  echo "=== Install T5Gemma dependencies ==="
  bash "${ROOT_DIR}/src/T5Gemma/install_deps.sh"
fi

bash "${ROOT_DIR}/src/T5Gemma/run_lrsum_pipeline.sh"
