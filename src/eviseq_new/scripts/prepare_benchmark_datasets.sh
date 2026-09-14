#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare the three long-form benchmark datasets into the canonical AFMR tree.
# By default the script expects INPUT_ROOT/{arxiv,booksum,govreport}.  Individual
# --*-input options override those paths when the raw releases are elsewhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
INPUT_ROOT="${INPUT_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/datasets}"
RAW_ROOT="${RAW_ROOT:-$ROOT/data/raw}"
DATASETS="${DATASETS:-arxiv,booksum,govreport}"
ARXIV_INPUT_DIR="${ARXIV_INPUT_DIR:-}"
BOOKSUM_INPUT_DIR="${BOOKSUM_INPUT_DIR:-}"
GOVREPORT_INPUT_DIR="${GOVREPORT_INPUT_DIR:-}"
ALLOW_CROSS_SPLIT_CONTENT="${ALLOW_CROSS_SPLIT_CONTENT:-0}"
DETOKENIZE_ARGS=()
ALLOW_DUPLICATE_IDS=0

usage() {
    cat <<'EOF'
Usage:
  prepare_benchmark_datasets.sh [options]

Options:
  --input-root DIR       Raw dataset root; uses DIR/arxiv, DIR/booksum, DIR/govreport
  --arxiv-input DIR      Raw ArXiv directory
  --booksum-input DIR    Raw BookSum directory
  --govreport-input DIR  Raw GovReport directory
  --output-root DIR      Canonical output root (default: ../datasets)
  --raw-root DIR         Raw-copy root (default: ../data/raw)
  --datasets LIST        Comma-separated subset (default: arxiv,booksum,govreport)
  --detokenize           Enable punctuation normalization for every selected dataset
  --no-detokenize        Disable punctuation normalization for every selected dataset
  --allow-duplicate-ids  Add a deterministic row suffix instead of failing within a split
  --allow-cross-split-content
  -h, --help

Examples:
  INPUT_ROOT=/data/summarization \
    PYTHON=/absolute/path/to/bienkieu_env/bin/python \
    bash scripts/prepare_benchmark_datasets.sh

  bash scripts/prepare_benchmark_datasets.sh \
    --arxiv-input /data/arxiv \
    --booksum-input /data/booksum/chapter-level \
    --govreport-input /data/gov_report \
    --datasets arxiv,booksum,govreport
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-root)
            [[ $# -ge 2 ]] || { echo "--input-root requires a directory" >&2; exit 2; }
            INPUT_ROOT="$2"
            shift 2
            ;;
        --arxiv-input)
            [[ $# -ge 2 ]] || { echo "--arxiv-input requires a directory" >&2; exit 2; }
            ARXIV_INPUT_DIR="$2"
            shift 2
            ;;
        --booksum-input)
            [[ $# -ge 2 ]] || { echo "--booksum-input requires a directory" >&2; exit 2; }
            BOOKSUM_INPUT_DIR="$2"
            shift 2
            ;;
        --govreport-input)
            [[ $# -ge 2 ]] || { echo "--govreport-input requires a directory" >&2; exit 2; }
            GOVREPORT_INPUT_DIR="$2"
            shift 2
            ;;
        --output-root)
            [[ $# -ge 2 ]] || { echo "--output-root requires a directory" >&2; exit 2; }
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --raw-root)
            [[ $# -ge 2 ]] || { echo "--raw-root requires a directory" >&2; exit 2; }
            RAW_ROOT="$2"
            shift 2
            ;;
        --datasets)
            [[ $# -ge 2 ]] || { echo "--datasets requires a comma-separated list" >&2; exit 2; }
            DATASETS="$2"
            shift 2
            ;;
        --detokenize)
            DETOKENIZE_ARGS=(--detokenize)
            shift
            ;;
        --no-detokenize)
            DETOKENIZE_ARGS=(--no-detokenize)
            shift
            ;;
        --allow-duplicate-ids)
            ALLOW_DUPLICATE_IDS=1
            shift
            ;;
        --allow-cross-split-content)
            ALLOW_CROSS_SPLIT_CONTENT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

input_for() {
    case "$1" in
        arxiv) printf '%s' "$ARXIV_INPUT_DIR" ;;
        booksum) printf '%s' "$BOOKSUM_INPUT_DIR" ;;
        govreport) printf '%s' "$GOVREPORT_INPUT_DIR" ;;
        *) return 1 ;;
    esac
}

IFS=',' read -r -a selected_datasets <<< "$DATASETS"
for dataset in "${selected_datasets[@]}"; do
    dataset="${dataset//[[:space:]]/}"
    [[ -n "$dataset" ]] || continue
    case "$dataset" in
        arxiv|booksum|govreport) ;;
        *) echo "Unsupported benchmark dataset: $dataset (choose arxiv, booksum, govreport)" >&2; exit 2 ;;
    esac

    input_dir="$(input_for "$dataset")"
    if [[ -z "$input_dir" && -n "$INPUT_ROOT" ]]; then
        input_dir="$INPUT_ROOT/$dataset"
    fi
    if [[ -z "$input_dir" ]]; then
        echo "No raw input configured for $dataset; set --${dataset}-input or --input-root" >&2
        exit 2
    fi

    command=(
        "$ROOT/scripts/prepare_afmr.sh"
        --dataset "$dataset"
        --input-dir "$input_dir"
        --output-dir "$OUTPUT_ROOT/$dataset"
        --raw-copy-dir "$RAW_ROOT/$dataset"
    )
    if [[ ${#DETOKENIZE_ARGS[@]} -gt 0 ]]; then
        command+=("${DETOKENIZE_ARGS[@]}")
    fi
    if [[ "$ALLOW_CROSS_SPLIT_CONTENT" == "1" || "$ALLOW_CROSS_SPLIT_CONTENT" == "true" || "$ALLOW_CROSS_SPLIT_CONTENT" == "yes" ]]; then
        command+=(--allow-cross-split-content)
    fi
    if [[ "$ALLOW_DUPLICATE_IDS" -eq 1 ]]; then
        command+=(--allow-duplicate-ids)
    fi
    echo "[$dataset] $input_dir -> $OUTPUT_ROOT/$dataset"
    PYTHON="$PYTHON_BIN" "${command[@]}"
done
