#!/usr/bin/env bash
set -Eeuo pipefail

# Load EviSeq PubMed epoch_004, train one additional full-finetune epoch,
# then evaluate the resulting last.pt on the test split with batch size 96.
# The original run directory is never overwritten.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
RUNNER="${PROJECT_ROOT}/scripts/run.sh"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" && -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -z "${PYTHON_BIN}" && -x "/workspace/storage-shared/nlp/dungdx4/bienkieu_env/bin/python" ]]; then
  PYTHON_BIN="/workspace/storage-shared/nlp/dungdx4/bienkieu_env/bin/python"
elif [[ -z "${PYTHON_BIN}" && -x "/Users/kieugiangbien/bienkieu_env/bin/python" ]]; then
  PYTHON_BIN="/Users/kieugiangbien/bienkieu_env/bin/python"
elif [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
export PYTHON_BIN

CHECKPOINT="${EVISEQ_CHECKPOINT:-${PROJECT_ROOT}/runs/eviseq/pubmed_qwen3_evidence/epoch_004.pt}"
SOURCE_RUN_DIR="$(cd "$(dirname "${CHECKPOINT}")" && pwd -P)"
SOURCE_CONFIG="${EVISEQ_SOURCE_CONFIG:-${SOURCE_RUN_DIR}/resolved_config.yaml}"
OUTPUT_DIR="${EVISEQ_CONTINUE_OUTPUT_DIR:-${PROJECT_ROOT}/runs/eviseq/pubmed_continue_epoch5}"
EVAL_BATCH_SIZE="${EVISEQ_EVAL_BATCH_SIZE:-96}"
TRAIN_BATCH_SIZE="${EVISEQ_TRAIN_BATCH_SIZE:-}"
PREDICTIONS="${OUTPUT_DIR}/epoch5_test_predictions.jsonl"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "ERROR: source resolved config not found: ${SOURCE_CONFIG}" >&2
  echo "Set EVISEQ_SOURCE_CONFIG to the resolved_config.yaml matching epoch_004.pt." >&2
  exit 1
fi

# Keep the exact training batch/model/data settings of the completed run.
# Only the stage duration and output directory are changed for the continuation.
RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eviseq-continue.XXXXXX")"
RUNTIME_CONFIG="${RUNTIME_DIR}/resolved_continue.yaml"
cleanup() { rm -rf "${RUNTIME_DIR}"; }
trap cleanup EXIT
"${PYTHON_BIN}" - "${SOURCE_CONFIG}" "${RUNTIME_CONFIG}" "${OUTPUT_DIR}" "${TRAIN_BATCH_SIZE}" <<'PY'
from pathlib import Path
import sys
import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
requested_batch_size = sys.argv[4].strip()
config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
config.setdefault("experiment", {})["name"] = "eviseq_pubmed_continue_epoch5"
config["experiment"]["output_dir"] = str(output_dir)
training = config.setdefault("training", {})
training["interface_warmup_epochs"] = 0
training["full_finetune_epochs"] = 1
if requested_batch_size:
    training["batch_size"] = int(requested_batch_size)
objectives = config.setdefault("objectives", {})
objectives["evidence_contrastive_warmup_epochs"] = 0
destination.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"source_config={source}")
print(f"train_batch_size={training.get('batch_size')}")
print(f"gradient_accumulation_steps={training.get('gradient_accumulation_steps')}")
PY

echo "=== EviSeq continuation: epoch_004 -> one full epoch ==="
echo "checkpoint: ${CHECKPOINT}"
echo "source config: ${SOURCE_CONFIG}"
echo "output:     ${OUTPUT_DIR}"
echo "eval batch: ${EVAL_BATCH_SIZE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  bash "${RUNNER}" train "${RUNTIME_CONFIG}" \
    --init-checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --overwrite-output-dir

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  bash "${RUNNER}" evaluate \
    "${OUTPUT_DIR}/resolved_config.yaml" \
    "${OUTPUT_DIR}/last.pt" \
    "${PREDICTIONS}" \
    --split test \
    --batch-size "${EVAL_BATCH_SIZE}"

METRICS="${OUTPUT_DIR}/epoch5_test_predictions.metrics.json"
if [[ -f "${METRICS}" ]]; then
  echo "=== Evaluation metrics ==="
  "${PYTHON_BIN}" -m json.tool "${METRICS}"
fi
