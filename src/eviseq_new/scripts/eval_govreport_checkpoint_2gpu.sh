#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLIT=test
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
RUN_DIR=""
CHECKPOINT="epoch_003.pt"
OUTPUT=""
ROUGE155_SCRIPT="${ROUGE155_SCRIPT:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_govreport_checkpoint_2gpu.sh --run-dir RUN_DIR [options]

Evaluate a GovReport checkpoint using two independent GPU shards, then merge
the predictions and compute the standard AFMR metrics.

Options:
  --run-dir PATH       Training run directory containing resolved_config.yaml
  --checkpoint PATH    Checkpoint path, or filename relative to RUN_DIR
                       (default: epoch_003.pt)
  --output PATH        Merged prediction JSONL (default: RUN_DIR/<checkpoint>_test_predictions.jsonl)
  --gpus IDS           Two GPU IDs, comma-separated (default: 0,1)
  --batch-size N       Eval batch size per GPU (default: 1)
  --split SPLIT        train, validation, or test (default: test)
  --rouge155 PATH      Optional ROUGE-1.5.5 evaluator script
  -h, --help           Show this message
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --run-dir) (($# >= 2)) || die "Missing value for --run-dir"; RUN_DIR="$2"; shift 2 ;;
    --checkpoint) (($# >= 2)) || die "Missing value for --checkpoint"; CHECKPOINT="$2"; shift 2 ;;
    --output) (($# >= 2)) || die "Missing value for --output"; OUTPUT="$2"; shift 2 ;;
    --gpus) (($# >= 2)) || die "Missing value for --gpus"; GPUS="$2"; shift 2 ;;
    --batch-size) (($# >= 2)) || die "Missing value for --batch-size"; BATCH_SIZE="$2"; shift 2 ;;
    --split) (($# >= 2)) || die "Missing value for --split"; SPLIT="$2"; shift 2 ;;
    --rouge155) (($# >= 2)) || die "Missing value for --rouge155"; ROUGE155_SCRIPT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "${RUN_DIR}" ]] || { usage >&2; die "--run-dir is required"; }
[[ -d "${RUN_DIR}" ]] || die "Run directory not found: ${RUN_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
CONFIG="${RUN_DIR}/resolved_config.yaml"
[[ -s "${CONFIG}" ]] || die "Missing resolved config: ${CONFIG}"

if [[ "${CHECKPOINT}" != /* && -f "${RUN_DIR}/${CHECKPOINT}" ]]; then
  CHECKPOINT="${RUN_DIR}/${CHECKPOINT}"
elif [[ "${CHECKPOINT}" != /* ]]; then
  CHECKPOINT="$(cd "$(dirname "${CHECKPOINT}")" 2>/dev/null && pwd)/$(basename "${CHECKPOINT}")" || \
    die "Checkpoint not found: ${CHECKPOINT}"
fi
[[ -s "${CHECKPOINT}" ]] || die "Checkpoint not found: ${CHECKPOINT}"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be a positive integer"
[[ "${SPLIT}" == train || "${SPLIT}" == validation || "${SPLIT}" == test ]] || \
  die "--split must be train, validation, or test"
[[ "${GPUS}" =~ ^[0-9]+,[0-9]+$ ]] || die "--gpus must contain exactly two GPU IDs, e.g. 0,1"
GPU0="${GPUS%%,*}"
GPU1="${GPUS##*,}"
[[ "${GPU0}" != "${GPU1}" ]] || die "--gpus must list two distinct GPU IDs"

if [[ -z "${OUTPUT}" ]]; then
  CHECKPOINT_STEM="$(basename "${CHECKPOINT}")"
  CHECKPOINT_STEM="${CHECKPOINT_STEM%.pt}"
  OUTPUT="${RUN_DIR}/${CHECKPOINT_STEM}_${SPLIT}_predictions.jsonl"
elif [[ "${OUTPUT}" != /* ]]; then
  OUTPUT="${RUN_DIR}/${OUTPUT}"
fi
OUTPUT_DIR="$(dirname "${OUTPUT}")"
mkdir -p "${OUTPUT_DIR}"
OUTPUT="${OUTPUT_DIR}/$(basename "${OUTPUT}")"
SHARD0="${OUTPUT%.jsonl}.shard0.jsonl"
SHARD1="${OUTPUT%.jsonl}.shard1.jsonl"
LOG0="${OUTPUT%.jsonl}.shard0.log"
LOG1="${OUTPUT%.jsonl}.shard1.log"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "Evaluating GovReport checkpoint: ${CHECKPOINT}"
echo "Split=${SPLIT}; GPUs=${GPU0},${GPU1}; batch/GPU=${BATCH_SIZE}"
echo "Predictions=${OUTPUT}"

CUDA_VISIBLE_DEVICES="${GPU0}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
  "${PYTHON:-python3}" "${ROOT}/run_afmr.py" evaluate \
  "${CONFIG}" "${CHECKPOINT}" "${SHARD0}" \
  --split "${SPLIT}" --batch-size "${BATCH_SIZE}" \
  --shard-rank 0 --num-shards 2 --device cuda:0 >"${LOG0}" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES="${GPU1}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
  "${PYTHON:-python3}" "${ROOT}/run_afmr.py" evaluate \
  "${CONFIG}" "${CHECKPOINT}" "${SHARD1}" \
  --split "${SPLIT}" --batch-size "${BATCH_SIZE}" \
  --shard-rank 1 --num-shards 2 --device cuda:0 >"${LOG1}" 2>&1 &
PID1=$!

FAILED=0
if ! wait "${PID0}"; then
  echo "GPU ${GPU0} evaluation failed; log: ${LOG0}" >&2
  tail -n 40 "${LOG0}" >&2
  FAILED=1
fi
if ! wait "${PID1}"; then
  echo "GPU ${GPU1} evaluation failed; log: ${LOG1}" >&2
  tail -n 40 "${LOG1}" >&2
  FAILED=1
fi
(( FAILED == 0 )) || die "Evaluation failed; shards were not merged"

"${PYTHON:-python3}" "${ROOT}/scripts/merge_eval_shards.py" --output "${OUTPUT}" "${SHARD0}" "${SHARD1}"

if [[ -z "${ROUGE155_SCRIPT}" ]]; then
  for candidate in "${ROOT}/../evaluation/evaluate_rouge.py" "${ROOT}/../rouge155/evaluate_rouge.py"; do
    if [[ -f "${candidate}" ]]; then
      ROUGE155_SCRIPT="${candidate}"
      break
    fi
  done
fi
if [[ -n "${ROUGE155_SCRIPT}" ]]; then
  [[ -f "${ROUGE155_SCRIPT}" ]] || die "ROUGE155_SCRIPT not found: ${ROUGE155_SCRIPT}"
  "${PYTHON:-python3}" "${ROUGE155_SCRIPT}" "${OUTPUT}" --output "${OUTPUT%.jsonl}.rouge155.json"
fi

echo "Evaluation complete: ${OUTPUT}"
echo "Shard logs: ${LOG0}, ${LOG1}"
