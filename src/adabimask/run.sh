#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" && -x "${HOME}/bienkieu_env/bin/python" ]]; then
  PYTHON_BIN="${HOME}/bienkieu_env/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_ROOT}/src/llm2seq${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
cd "${PROJECT_ROOT}"

COMMAND="${1:-train}"
if [[ $# -gt 0 ]]; then shift; fi

case "${COMMAND}" in
  train) exec "${PYTHON_BIN}" -m adabimask.training "$@" ;;
  eval) exec "${PYTHON_BIN}" -m adabimask.evaluate "$@" ;;
  export) exec "${PYTHON_BIN}" -m adabimask.export_policy "$@" ;;
  ablate) exec "${PYTHON_BIN}" -m adabimask.run_ablations "$@" ;;
  test) exec "${PYTHON_BIN}" "${SCRIPT_DIR}/tests/run_tests.py" "$@" ;;
  *) echo "Unknown command: ${COMMAND} (train|eval|export|ablate|test)" >&2; exit 2 ;;
esac
