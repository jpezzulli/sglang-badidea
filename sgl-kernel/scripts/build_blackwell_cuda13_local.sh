#!/usr/bin/env bash
# Local scratch helper for building sglang-kernel on Blackwell workstations with
# PyTorch cu130.  It deliberately keeps the CUDA toolkit seen by CMake aligned
# to Torch's CUDA minor (13.0) instead of whatever newer nvcc wheel may already
# be installed in the venv.
set -euo pipefail

VENV=${VENV:-/opt/bigballs/.venv}
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-${VENV}/bin/python}

# Match torch.version.cuda == 13.0.  Override only if the local torch wheel moves.
CUDA_PYPI_VERSION=${CUDA_PYPI_VERSION:-13.0.88}
BUILD_JOBS=${BUILD_JOBS:-4}
NVCC_THREADS=${NVCC_THREADS:-4}

if [[ ! -x "${PYTHON}" ]]; then
  echo "error: Python not found or not executable: ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" - <<'PY'
import torch
if torch.version.cuda != "13.0":
    raise SystemExit(f"expected torch.version.cuda == 13.0, got {torch.version.cuda!r}")
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
PY

# The unsuffixed NVIDIA CUDA wheel names are used intentionally because this
# environment already has nvidia-cuda-nvcc==13.3.73 from that namespace.
"${PYTHON}" -m pip install --force-reinstall --no-deps \
  "nvidia-cuda-nvcc==${CUDA_PYPI_VERSION}" \
  "nvidia-cuda-runtime==${CUDA_PYPI_VERSION}" \
  "nvidia-cuda-crt==${CUDA_PYPI_VERSION}" \
  "nvidia-nvvm==${CUDA_PYPI_VERSION}"

export CUDA_HOME=${CUDA_HOME:-$(${PYTHON} - <<'PY'
import pathlib, site
for base in site.getsitepackages():
    p = pathlib.Path(base) / "nvidia" / "cu13"
    if p.exists():
        print(p)
        break
else:
    raise SystemExit("could not find nvidia/cu13 in site-packages")
PY
)}
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--allow-unsupported-compiler}"

# Some NVIDIA wheel layouts use lib/ rather than lib64/, and libcudart.so.13
# rather than an unversioned linker soname.
ln -sfn lib "${CUDA_HOME}/lib64"
if [[ -e "${CUDA_HOME}/lib/libcudart.so.13" && ! -e "${CUDA_HOME}/lib/libcudart.so" ]]; then
  ln -s libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so"
fi

nvcc --version
rm -rf "${REPO_ROOT}/sgl-kernel/build" "${REPO_ROOT}/sgl-kernel/dist"
cd "${REPO_ROOT}/sgl-kernel"
CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" \
  "${PYTHON}" -m pip wheel . -w dist --no-build-isolation --no-deps \
  --config-settings=cmake.define.SGL_KERNEL_COMPILE_THREADS="${NVCC_THREADS}" \
  --config-settings=cmake.define.ENABLE_BELOW_SM90=OFF
"${PYTHON}" -m pip install --force-reinstall --no-deps dist/sglang_kernel-*.whl
