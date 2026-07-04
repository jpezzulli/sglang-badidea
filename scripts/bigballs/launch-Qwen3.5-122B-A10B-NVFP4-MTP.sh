#!/usr/bin/env bash
# unoptimized launcher
set -euo pipefail
export FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-24}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-24}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-24}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-24}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--allow-unsupported-compiler}"
#MODEL_PATH="${MODEL_PATH:-/srv/models/hf/bigballs}"
#PORT="${PORT:-8001}"
#SGLANG_BIN="${SGLANG_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/sglang}}"
#SGLANG_BIN="${SGLAN}"
#PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
#PYTHON_BIN="${PYTHON_BIN:-python}"
/opt/bigballs/.venv/bin/sglang serve \
  --model-path /srv/models/hf/bigballs \
  --load-format safetensors \
  --quantization modelopt_fp4 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 5 \
  --max-running-requests 1 \
  --max-total-tokens 393216 \
  --page-size 64 \
  --host 0.0.0.0 \
  --port 8001 \
  --tp 1 \
  --context-length 196608 \
  --mem-fraction-static 0.93 \
  --chunked-prefill-size 4096 \
  --attention-backend triton \
  --sampling-backend flashinfer \
  --watchdog-timeout 1800 \
  --served-model-name pennyroyal \
  --speculative-algo NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
