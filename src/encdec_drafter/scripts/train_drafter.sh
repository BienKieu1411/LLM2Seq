#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$PACKAGE_DIR/configs/t5gemma2.yaml}"
export PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m encdec_drafter train --config "$CONFIG" "$@"
