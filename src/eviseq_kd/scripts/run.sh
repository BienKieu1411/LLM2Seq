#!/usr/bin/env bash
set -Eeuo pipefail

# EviSeq-KD launcher. Run this file from any directory:
#   bash src/eviseq_kd/scripts/run.sh wiki --overwrite-output-dir

CALLER_CWD="$(pwd -P)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SRC_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd -P)"
RUNNER="${PROJECT_ROOT}/run.py"
cd "${PROJECT_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "/workspace/storage-shared/nlp/dungdx4/bienkieu_env/bin/python" ]]; then
  PYTHON_BIN="/workspace/storage-shared/nlp/dungdx4/bienkieu_env/bin/python"
elif [[ -x "/Users/kieugiangbien/bienkieu_env/bin/python" ]]; then
  PYTHON_BIN="/Users/kieugiangbien/bienkieu_env/bin/python"
else
  PYTHON_BIN="python3"
fi

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

WIKI_CONFIG="${PROJECT_ROOT}/configs/wikilingua_kd.yaml"
PUBMED_CONFIG="${PROJECT_ROOT}/configs/pubmed_kd.yaml"
SMOKE_CONFIG="${PROJECT_ROOT}/configs/smoke_a100.yaml"

absolute_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s\n' "${CALLER_CWD}/$1"
  fi
}

config_path() {
  case "${1:-}" in
    wiki|wikilingua) printf '%s\n' "${WIKI_CONFIG}" ;;
    pubmed) printf '%s\n' "${PUBMED_CONFIG}" ;;
    smoke|smoke-a100) printf '%s\n' "${SMOKE_CONFIG}" ;;
    *) absolute_path "${1:?Pass wiki, pubmed, smoke, or a YAML path}" ;;
  esac
}

config_value() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
from eviseq_kd.student.configuration import load_config

value = load_config(sys.argv[1])
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

run_train() {
  local config="$1"
  shift
  "${PYTHON_BIN}" "${RUNNER}" --config "${config}" "$@"
}

run_eval() {
  local config="$1"
  local checkpoint="$2"
  local output="$3"
  shift 3
  "${PYTHON_BIN}" "${PROJECT_ROOT}/evaluate.py" \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --output "${output}" \
    "$@"
}

evaluate_test_for_config() {
  local config="$1"
  shift
  local output_dir
  output_dir="$(config_value "${config}" experiment.output_dir)"
  if [[ "${output_dir}" != /* ]]; then
    output_dir="${PROJECT_ROOT}/${output_dir}"
  fi
  local resolved="${output_dir}/resolved_config.yaml"
  local checkpoint="${output_dir}/${CHECKPOINT_NAME:-last.pt}"
  local output="${output_dir}/${OUTPUT_NAME:-last_test_predictions.jsonl}"
  [[ -f "${resolved}" ]] || { echo "ERROR: missing ${resolved}" >&2; return 1; }
  [[ -f "${checkpoint}" ]] || { echo "ERROR: missing ${checkpoint}" >&2; return 1; }
  run_eval "${resolved}" "${checkpoint}" "${output}" --split test "$@"
}

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "${MODE}" in
  train)
    CONFIG="$(config_path "${1:?Pass a config alias or YAML path}")"
    shift
    run_train "${CONFIG}" "$@"
    ;;
  wiki|wikilingua)
    run_train "${WIKI_CONFIG}" "$@"
    ;;
  pubmed)
    run_train "${PUBMED_CONFIG}" "$@"
    ;;
  smoke|smoke-a100)
    run_train "${SMOKE_CONFIG}" "$@"
    ;;
  evaluate)
    CONFIG="$(absolute_path "${1:?Pass a resolved config YAML}")"
    CHECKPOINT="$(absolute_path "${2:?Pass a checkpoint}")"
    OUTPUT="$(absolute_path "${3:?Pass an output JSONL path}")"
    shift 3
    run_eval "${CONFIG}" "${CHECKPOINT}" "${OUTPUT}" "$@"
    ;;
  evaluate-wiki-test)
    evaluate_test_for_config "${WIKI_CONFIG}" "$@"
    ;;
  evaluate-pubmed-test)
    evaluate_test_for_config "${PUBMED_CONFIG}" "$@"
    ;;
  evaluate-test)
    CONFIG="$(config_path "${1:-wiki}")"
    if [[ $# -gt 0 ]]; then shift; fi
    evaluate_test_for_config "${CONFIG}" "$@"
    ;;
  rouge155)
    PREDICTIONS="$(absolute_path "${1:?Pass predictions JSONL}")"
    shift
    "${PYTHON_BIN}" "${SRC_ROOT}/rouge155/evaluate_rouge.py" "${PREDICTIONS}" "$@"
    ;;
  test)
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -m pytest -q "${PROJECT_ROOT}/tests"
    ;;
  inspect)
    CONFIG="$(config_path "${1:-wiki}")"
    config_value "${CONFIG}" experiment.name
    config_value "${CONFIG}" experiment.output_dir
    config_value "${CONFIG}" training.distillation.mode
    config_value "${CONFIG}" training.batch_size
    config_value "${CONFIG}" training.gradient_accumulation_steps
    ;;
  *)
    cat <<'EOF'
EviSeq-KD (online Qwen3-4B teacher)

  bash src/eviseq_kd/scripts/run.sh train wiki --overwrite-output-dir
  bash src/eviseq_kd/scripts/run.sh pubmed --overwrite-output-dir
  bash src/eviseq_kd/scripts/run.sh smoke --overwrite-output-dir
  bash src/eviseq_kd/scripts/run.sh evaluate-pubmed-test --batch-size 8
  bash src/eviseq_kd/scripts/run.sh evaluate-test wiki --batch-size 8
  bash src/eviseq_kd/scripts/run.sh rouge155 runs/.../last_test_predictions.jsonl --details
  bash src/eviseq_kd/scripts/run.sh test
  bash src/eviseq_kd/scripts/run.sh inspect pubmed

The paper configs use training.distillation.mode=online. They do not build or
read a teacher cache. Set PYTHON_BIN explicitly when the server environment is
not activated; otherwise the launcher uses the active virtualenv or the shared
bienkieu_env path.
EOF
    ;;
esac
