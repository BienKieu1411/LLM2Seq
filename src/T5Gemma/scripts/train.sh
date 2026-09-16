#!/usr/bin/env bash
set -euo pipefail

# Preserve caller overrides while load_env.sh reads the optional env file.
# This lets the PubMed wrapper select a config without editing env.txt.
CONFIG_OVERRIDE=""
if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Usage: $0 [--config CONFIG]" >&2
    exit 2
  fi
  CONFIG_OVERRIDE="$2"
  shift 2
fi
if [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--config CONFIG]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

if [[ -n "${CONFIG_OVERRIDE}" ]]; then
  CONFIG="${CONFIG_OVERRIDE}"
  export CONFIG
fi

visible_gpu_count=1
if [[ -n "${T5GEMMA_NPROC_PER_NODE:-}" ]]; then
  visible_gpu_count="${T5GEMMA_NPROC_PER_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
  IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  visible_gpu_count="${#visible_gpus[@]}"
fi
if [[ ! "${visible_gpu_count}" =~ ^[12]$ ]]; then
  echo "T5Gemma supports one or two GPUs; got nproc_per_node=${visible_gpu_count}." >&2
  exit 2
fi
if [[ -n "${T5GEMMA_NPROC_PER_NODE:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
  IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  if (( visible_gpu_count > ${#visible_gpus[@]} )); then
    echo "nproc_per_node=${visible_gpu_count} exceeds the ${#visible_gpus[@]} visible CUDA devices." >&2
    exit 2
  fi
fi

mkdir -p "${LOG_DIR}"
ts="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/${ts}_train_full.log"

echo "=== T5Gemma full fine-tune ==="
echo "Config: ${CONFIG}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all visible devices}"
echo "Processes: ${visible_gpu_count} (DDP when two GPUs are selected)"
echo "Log: ${log_file}"

train_args=(--config "${CONFIG}")
case "${OVERWRITE_OUTPUT_DIR:-false}" in
  true|TRUE|yes|YES|1)
    train_args+=(--overwrite-output-dir)
    ;;
esac

if [[ "${visible_gpu_count}" == "2" ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    "${T5GEMMA_ROOT}/scripts/train_full.py" \
    "${train_args[@]}" \
    2>&1 | tee "${log_file}"
else
  "${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/train_full.py" \
    "${train_args[@]}" \
    2>&1 | tee "${log_file}"
fi
