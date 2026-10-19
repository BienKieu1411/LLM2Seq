#!/usr/bin/env bash
set -euo pipefail
SI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "${PYTHON_BIN:-python3}" "$SI_ROOT/prepare_si.py" "$@"
