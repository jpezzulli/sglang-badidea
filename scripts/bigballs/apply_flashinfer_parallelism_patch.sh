#!/usr/bin/env bash
# Scratch/local blackhouse helper: patch installed FlashInfer and existing cache files.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-24}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-4}"
python "$SCRIPT_DIR/patch_flashinfer_parallelism.py"
python "$SCRIPT_DIR/patch_flashinfer_cache_build_ninja.py" --nvcc-threads "$FLASHINFER_NVCC_THREADS"
