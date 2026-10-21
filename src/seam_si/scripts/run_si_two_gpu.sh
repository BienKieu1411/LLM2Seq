#!/usr/bin/env bash
set -euo pipefail
SI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Compatibility alias; GPU_IDS/CUDA_VISIBLE_DEVICES can now name any count.
exec "$SI_ROOT/scripts/run_si_multi_gpu.sh" "$@"
