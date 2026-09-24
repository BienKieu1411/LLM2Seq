#!/usr/bin/env bash
set -Eeuo pipefail

export T5GEMMA_DATASET=booksum
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_longform.sh" "$@"
