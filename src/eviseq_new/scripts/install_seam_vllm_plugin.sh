#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
python_bin="${SEAM_VLLM_PYTHON:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "${python_bin}" >&2
  exit 1
fi

"${python_bin}" -m pip install --no-deps --no-build-isolation --editable "${project_dir}"
"${python_bin}" -c 'import vllm; print("vLLM", vllm.__version__)'
printf '\nInstalled the local SEAM vLLM plugin without changing package dependencies.\n'
printf 'Activate the environment that contains vLLM 0.25.1 before running this script.\n'
