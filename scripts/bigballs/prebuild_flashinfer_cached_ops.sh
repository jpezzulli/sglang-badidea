#!/usr/bin/env bash
# Scratch/local blackhouse helper: prebuild existing FlashInfer cached ops.
set -euo pipefail
export FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-24}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-4}"
CACHE_ROOT="${FLASHINFER_CACHE_ROOT:-$HOME/.cache/flashinfer}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/patch_flashinfer_cache_build_ninja.py" --cache-root "$CACHE_ROOT" --nvcc-threads "$FLASHINFER_NVCC_THREADS"
mapfile -t builds < <(find "$CACHE_ROOT" -name build.ninja -type f -printf '%h\n' 2>/dev/null | sort -u)
if (( ${#builds[@]} == 0 )); then echo "No FlashInfer build.ninja files under $CACHE_ROOT"; exit 0; fi
for d in "${builds[@]}"; do
  echo "==> ninja -C $d -j$FLASHINFER_NINJA_JOBS (nvcc threads patched to $FLASHINFER_NVCC_THREADS)"
  ninja -C "$d" -j"$FLASHINFER_NINJA_JOBS"
done
