#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_govreport.sh" "$@"
