#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIENKIEU_PYTHON="/Users/kieugiangbien/bienkieu_env/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "${BIENKIEU_PYTHON}" ]]; then
  PYTHON_BIN="${BIENKIEU_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
cd "${SCRIPT_DIR}"

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
