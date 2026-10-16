#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
export PYTHONPATH="${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_USE_V2_MODEL_RUNNER=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "${SEAM_VLLM_PYTHON:-python3}" -m seam_vllm.serve "$@"
