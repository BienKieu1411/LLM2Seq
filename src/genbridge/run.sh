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
  train) exec "${PYTHON_BIN}" -m genbridge.training "$@" ;;
  eval) exec "${PYTHON_BIN}" -m genbridge.evaluate "$@" ;;
  ablate) exec "${PYTHON_BIN}" -m genbridge.run_ablations "$@" ;;
  prepare-wikilingua) exec "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_wikilingua.py" "$@" ;;
  prepare-lrsum) exec "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_lrsum.py" "$@" ;;
  count-params) exec "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/count_parameters.py" "$@" ;;
  test) exec "${PYTHON_BIN}" "${SCRIPT_DIR}/tests/run_tests.py" "$@" ;;
  *) echo "Unknown command: ${COMMAND} (train|eval|ablate|prepare-wikilingua|prepare-lrsum|count-params|test)" >&2; exit 2 ;;
esac
