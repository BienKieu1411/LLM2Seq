#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${ROOT_DIR}/src/llm2seq/scripts/run_lrsum_llama.sh"
bash "${ROOT_DIR}/src/llm2seq/scripts/run_lrsum_qwen.sh"
bash "${ROOT_DIR}/src/T5Gemma/run_lrsum_pipeline.sh"
