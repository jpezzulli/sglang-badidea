#!/usr/bin/env bash
# Scratch/local blackhouse helper: patch installed FlashInfer and existing cache files.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-24}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" "$SCRIPT_DIR/patch_flashinfer_parallelism.py"
"$PYTHON_BIN" "$SCRIPT_DIR/patch_flashinfer_cache_build_ninja.py" --nvcc-threads "$FLASHINFER_NVCC_THREADS"
