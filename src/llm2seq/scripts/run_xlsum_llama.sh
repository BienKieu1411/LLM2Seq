#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_VARIANT=llama bash "${SCRIPT_DIR}/run_xlsum_pipeline.sh"
