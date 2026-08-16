#!/bin/bash
# =============================================================================
# Real-World N-Day Vulnerability Testing Pipeline — End-to-End Runner
# =============================================================================
# Usage:
#   bash run_pipeline.sh --project ffmpeg --weights /path/to/model_last.pt
#   bash run_pipeline.sh --test     # Quick self-test with synthetic data
#
# This script runs the full pipeline:
#   1. Extract functions from source repo
#   2. Generate USCP graphs via ATLAS
#   3. Run model inference
#   4. Print detection report
# =============================================================================

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT=""
WEIGHTS=""
SOURCE_DIR=""
OUTPUT_DIR="$SCRIPT_DIR/output"
LIMIT=""
TEST_MODE=false
LANG="c"

# External drive base path
EXTERNAL_BASE="/media/user1/One Touch1/00 Data/realdatavul"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)   PROJECT="$2";    shift 2 ;;
        --weights)   WEIGHTS="$2";    shift 2 ;;
        --source_dir) SOURCE_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --limit)     LIMIT="$2";      shift 2 ;;
        --lang)      LANG="$2";       shift 2 ;;
        --test)      TEST_MODE=true;  shift ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_pipeline.sh --project <name> --weights <path> [--source_dir <path>] [--limit N] [--test]"
            exit 1
            ;;
    esac
done

# ── Test mode ─────────────────────────────────────────────────────────────────
if [ "$TEST_MODE" = true ]; then
    echo "==========================================================="
    echo "  Running Pipeline Self-Test (synthetic data)"
    echo "==========================================================="
    cd "$SCRIPT_DIR"
    python test_pipeline.py
    exit $?
fi

# ── Validate arguments ────────────────────────────────────────────────────────
if [ -z "$PROJECT" ]; then
    echo "ERROR: --project is required (linux, openssl, ffmpeg, qemu, xen, libav)"
    exit 1
fi

# Auto-detect source directory from project name
if [ -z "$SOURCE_DIR" ]; then
    case "$PROJECT" in
        linux)   SOURCE_DIR="$EXTERNAL_BASE/linux_versions/linux-4.15.1" ;;
        openssl) SOURCE_DIR="$EXTERNAL_BASE/openssl" ;;
        ffmpeg)  SOURCE_DIR="$EXTERNAL_BASE/FFmpeg" ;;
        qemu)    SOURCE_DIR="$EXTERNAL_BASE/qemu" ;;
        xen)     SOURCE_DIR="$EXTERNAL_BASE/xen" ;;
        libav)   SOURCE_DIR="$EXTERNAL_BASE/libav" ;;
        *)
            echo "ERROR: Unknown project '$PROJECT'. Provide --source_dir manually."
            exit 1
            ;;
    esac
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Setup output directory
PROJ_OUTPUT="$OUTPUT_DIR/$PROJECT"
mkdir -p "$PROJ_OUTPUT"

# Build optional arguments
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

WEIGHTS_ARG=""
if [ -n "$WEIGHTS" ]; then
    WEIGHTS_ARG="--weights_path $WEIGHTS"
fi

echo "==========================================================="
echo "  Real-World N-Day Vulnerability Testing Pipeline"
echo "==========================================================="
echo "  Project    : $PROJECT"
echo "  Source Dir : $SOURCE_DIR"
echo "  Output Dir : $PROJ_OUTPUT"
echo "  Weights    : ${WEIGHTS:-'(none — random weights)'}"
echo "  Limit      : ${LIMIT:-'(all functions)'}"
echo "==========================================================="

# ── Step 1: Extract functions ─────────────────────────────────────────────────
echo ""
echo "[1/3] Extracting C/C++ functions from $PROJECT..."
cd "$SCRIPT_DIR"
python extract_functions.py \
    --source_dir "$SOURCE_DIR" \
    --project "$PROJECT" \
    --output "$PROJ_OUTPUT/functions.jsonlines" \
    --lang "$LANG" \
    $LIMIT_ARG

# Check if any functions were extracted
FUNC_COUNT=$(wc -l < "$PROJ_OUTPUT/functions.jsonlines")
if [ "$FUNC_COUNT" -eq 0 ]; then
    echo "ERROR: No functions extracted. Check source directory."
    exit 1
fi
echo "  → Extracted $FUNC_COUNT functions"

# ── Step 2: Generate USCP graphs ─────────────────────────────────────────────
echo ""
echo "[2/3] Generating USCP graphs via ATLAS..."
python generate_graphs.py \
    --input "$PROJ_OUTPUT/functions.jsonlines" \
    --output "$PROJ_OUTPUT/test_uscp.jsonlines" \
    --lang "$LANG" \
    $LIMIT_ARG

GRAPH_COUNT=$(wc -l < "$PROJ_OUTPUT/test_uscp.jsonlines")
echo "  → Generated graphs for $GRAPH_COUNT functions"

# ── Step 3: Run model inference ───────────────────────────────────────────────
echo ""
echo "[3/3] Running UCG-VD model inference..."
python infer_nday.py \
    --test_data "$PROJ_OUTPUT/test_uscp.jsonlines" \
    --output_dir "$PROJ_OUTPUT/results" \
    $WEIGHTS_ARG

echo ""
echo "==========================================================="
echo "  Pipeline Complete!"
echo "  Results: $PROJ_OUTPUT/results/"
echo "  Report:  $PROJ_OUTPUT/results/nday_report.txt"
echo "==========================================================="
