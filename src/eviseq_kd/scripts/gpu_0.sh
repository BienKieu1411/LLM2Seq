#!/usr/bin/env bash
set -Eeuo pipefail

# One-command server queue for EviSeq-KD.
# Default: online-Qwen3-4B teacher on PubMed, then test evaluation.
# Examples:
#   CUDA_VISIBLE_DEVICES=0 bash src/eviseq_kd/scripts/gpu_0.sh
#   KD_TASK=wiki EVAL_BATCH_SIZE=16 CUDA_VISIBLE_DEVICES=0 \
#     bash src/eviseq_kd/scripts/gpu_0.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
SRC_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd -P)"
cd "${SRC_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TASK="${KD_TASK:-pubmed}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-last.pt}"
OUTPUT_NAME="${OUTPUT_NAME:-${CHECKPOINT_NAME%.pt}_test_predictions.jsonl}"
case "${TASK}" in
  wiki|wikilingua) RUN_DIR_NAME="wikilingua_qwen3_4b_teacher" ;;
  pubmed) RUN_DIR_NAME="pubmed_qwen3_4b_teacher" ;;
  smoke|smoke-a100) RUN_DIR_NAME="smoke_a100" ;;
  *) echo "ERROR: KD_TASK must be wiki, pubmed, or smoke" >&2; exit 2 ;;
esac
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/gpu_0_${TASK}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== EviSeq-KD online queue ==="
echo "task=${TASK} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} eval_batch=${EVAL_BATCH_SIZE}"
echo "log=${LOG_FILE}"

bash "${PROJECT_ROOT}/scripts/run.sh" "${TASK}" --overwrite-output-dir

CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
OUTPUT_NAME="${OUTPUT_NAME}" \
bash "${PROJECT_ROOT}/scripts/run.sh" "evaluate-${TASK}-test" \
  --batch-size "${EVAL_BATCH_SIZE}"

PREDICTIONS="${PROJECT_ROOT}/runs/eviseq_kd/${RUN_DIR_NAME}/${OUTPUT_NAME}"
if [[ -n "${PYROUGE_HOME_DIR:-}" && -f "${PYROUGE_HOME_DIR}/ROUGE-1.5.5.pl" ]]; then
  bash "${PROJECT_ROOT}/scripts/run.sh" rouge155 "${PREDICTIONS}" --details
else
  echo "PYROUGE_HOME_DIR is unset or invalid; skipping Perl ROUGE-1.5.5."
fi

echo "=== EviSeq-KD queue completed ==="
