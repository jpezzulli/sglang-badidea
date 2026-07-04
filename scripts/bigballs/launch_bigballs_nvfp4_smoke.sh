#!/usr/bin/env bash
# Scratch/local blackhouse helper: launch the local NVFP4 smoke-test server.
set -euo pipefail
export FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-24}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-24}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-24}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-24}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--allow-unsupported-compiler}"
MODEL_PATH="${MODEL_PATH:-/srv/models/hf/bigballs}"
PORT="${PORT:-8011}"
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --load-format safetensors \
  --quantization modelopt_fp4 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp 1 \
  --context-length 32768 \
  --mem-fraction-static 0.82 \
  --chunked-prefill-size 4096 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-flashinfer-autotune \
  --disable-cuda-graph \
  --skip-server-warmup \
  --language-only \
  --watchdog-timeout 1800 \
  --served-model-name bigballs-nvfp4
