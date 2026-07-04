# Local Blackwell CUDA 13 sglang-kernel build notes

This scratch branch targets a Fedora workstation with an RTX PRO 6000 Blackwell
Workstation GPU, Python 3.14 under `/opt/bigballs/.venv`, and a PyTorch nightly
whose `torch.version.cuda` is `13.0`.

## Diagnosis

`sgl-kernel/CMakeLists.txt` enables CUDA, finds `CUDAToolkit`, and then imports
Torch with `find_package(Torch REQUIRED)`. Torch's CMake package performs its own
CUDA consistency check. If the venv has `nvidia-cuda-nvcc==13.3.73`, CMake sees
nvcc/CUDA 13.3 while the Torch cu130 headers report CUDA 13.0, so configure fails
before any SGLang kernel source is compiled.

Treating nvcc 13.3 as "close enough" is the uglier option: it requires patching
or shadowing PyTorch's CMake checks, and it would build extensions against a CUDA
minor newer than the runtime ABI Torch says it was built for. For this local
experiment the least ugly reversible fix is to make the venv's CUDA compiler
minor match Torch cu130, then keep the existing SGLang CUDA 13 code paths.

## Recommended command recipe

From the repository root:

```bash
export VENV=/opt/bigballs/.venv
export CUDA_PYPI_VERSION=13.0.88
export BUILD_JOBS=4
export NVCC_THREADS=4
sgl-kernel/scripts/build_blackwell_cuda13_local.sh
```

The script does the following:

1. Verifies `torch.version.cuda == "13.0"`.
2. Force-reinstalls the local NVIDIA CUDA compiler/runtime wheel set at
   `13.0.88` so `nvcc --version` and Torch agree on CUDA 13.0.
3. Points `CUDA_HOME`, `PATH`, and `LD_LIBRARY_PATH` at the venv CUDA tree.
4. Recreates the local wheel-layout symlinks:
   - `$CUDA_HOME/lib64 -> lib`
   - `$CUDA_HOME/lib/libcudart.so -> libcudart.so.13`
5. Builds from `sgl-kernel/` with no build isolation, conservative local
   parallelism, and `ENABLE_BELOW_SM90=OFF` to avoid wasting time on pre-Hopper
   code generation for the Blackwell-only workstation experiment.
6. Installs the freshly built `dist/sglang_kernel-*.whl` back into the venv.

## Wheel fallback

Before building from source, it is reasonable to try the existing CUDA 13 wheel
for this exact package version:

```bash
/opt/bigballs/.venv/bin/python -m pip install --force-reinstall --no-deps \
  --index-url https://docs.sglang.ai/whl/cu130/ \
  sglang-kernel==0.4.4
```

That is faster and cleaner if it imports and covers the required Blackwell paths.
Use the source build above when the prebuilt wheel lacks the needed SM 12.x code
or experiments require local kernel changes.

## Explicit non-recommendation

Do not downgrade Torch for this branch unless CUDA 13.0 nvcc/runtime wheels are
unavailable in the target venv. The current failure is a toolkit-minor mismatch,
not evidence that Torch 2.14.0.dev+cu130 itself is unusable.
