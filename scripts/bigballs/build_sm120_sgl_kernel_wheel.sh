#!/usr/bin/env bash
# Scratch/local blackhouse helper: build an SM120-only sglang-kernel wheel.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export MAX_JOBS="${MAX_JOBS:-24}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-24}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--allow-unsupported-compiler}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
cd "$REPO_ROOT/sgl-kernel"
./scripts/apply_bigballs_sm120_only_patch.sh
rm -rf dist build
"$PYTHON_BIN" -m pip wheel . -w dist --no-build-isolation --no-deps
printf '\nBuilt wheels:\n'
find dist -maxdepth 1 -type f -name '*.whl' -print | sort
