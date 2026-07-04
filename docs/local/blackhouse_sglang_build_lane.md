# Blackhouse local SGLang / FlashInfer build lane

This page codifies the scratch/local `bigballs-cuda13x-frisky` lane for the
blackhouse RTX PRO 6000 Blackwell workstation. It is not a production deployment
recipe and must not touch production `llama.cpp`, `llmbrain.service`, model
service units, or global NVIDIA wheel policy.

## Scope and assumptions

- Host GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (`SM120`).
- OS/toolchain: Fedora with GCC 16.
- Python: the `/opt/bigballs` Python 3.14 virtual environment.
- Torch: nightly CUDA 13.0 (`cu130`).
- CUDA userland: CUDA 13.x wheel layout, mostly aligned to 13.3 where available.
- Model: `/srv/models/hf/bigballs`, from `txn545/Qwen3.5-122B-A10B-NVFP4`.
- SGLang branch: `bigballs-cuda13x-frisky`.
- The prebuilt `sglang-kernel` `cu130` wheel is not assumed to work; build the
  local wheel from this checkout.

Do not run `pip install sglang[all]`, do not mass-upgrade NVIDIA wheels, and do
not commit generated FlashInfer caches, built wheels, model files, virtualenv
contents, or logs.

## Default environment

Activate the local environment first, then export the build-lane defaults:

```bash
source /opt/bigballs/bin/activate
export FLASHINFER_NINJA_JOBS=24
export FLASHINFER_NVCC_THREADS=4
export MAX_JOBS=24
export CMAKE_BUILD_PARALLEL_LEVEL=24
export TORCHINDUCTOR_COMPILE_THREADS=24
export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"
```

`FLASHINFER_NINJA_JOBS` controls host-side Ninja parallelism for FlashInfer cache
builds. `FLASHINFER_NVCC_THREADS` controls the `nvcc --threads` value emitted
into generated cached-op `build.ninja` files.

## Build and install the local SM120-only `sgl-kernel` wheel

From the repository root:

```bash
PYTHON_BIN=/opt/bigballs/bin/python \
  scripts/bigballs/build_sm120_sgl_kernel_wheel.sh

PYTHON_BIN=/opt/bigballs/bin/python \
  scripts/bigballs/install_check_sgl_kernel_wheel.sh
```

The build script intentionally applies the existing scratch-only
`sgl-kernel/scripts/apply_bigballs_sm120_only_patch.sh` helper before building.
That helper narrows CUDA codegen to the SM120 path, disables the duplicate
`common_ops_sm90` target, disables FlashMLA/FA3 overbuild for this probe, and
keeps the CMake edits local to the working tree. Do not commit those generated
CMake working-tree edits if the helper already applies them.

## Patch FlashInfer generated cached-op parallelism

Patch the installed FlashInfer package in the active Python environment:

```bash
/opt/bigballs/bin/python scripts/bigballs/patch_flashinfer_parallelism.py
```

The patch targets installed FlashInfer Python generator/source files that emit
hard-coded single-thread `nvcc --threads=1` or equivalent arguments. Future
cached-op `build.ninja` files should then use the environment-controlled values
instead of defaulting to single-thread CUDA compilation.

To patch already-generated cache files under `~/.cache/flashinfer/**/build.ninja`:

```bash
/opt/bigballs/bin/python scripts/bigballs/patch_flashinfer_cache_build_ninja.py
```

The combined wrapper applies both steps:

```bash
scripts/bigballs/apply_flashinfer_parallelism_patch.sh
```

## Prebuild existing FlashInfer cached ops

After the server has generated cached-op directories, or after reproducing a
runtime compile once, prebuild every existing FlashInfer cached-op directory with
aggressive local defaults:

```bash
scripts/bigballs/prebuild_flashinfer_cached_ops.sh
```

The helper patches existing `build.ninja` files first, then runs `ninja -j24` in
each cache directory by default. Override with `FLASHINFER_CACHE_ROOT`,
`FLASHINFER_NINJA_JOBS`, or `FLASHINFER_NVCC_THREADS` if needed.

## Launch the NVFP4 smoke server

Use the smoke launcher for the known-good shape:

```bash
scripts/bigballs/launch_bigballs_nvfp4_smoke.sh
```

The launcher serves `/srv/models/hf/bigballs` on `0.0.0.0:8011` as
`bigballs-nvfp4`, uses ModelOpt FP4/NVFP4, disables FlashInfer autotune and CUDA
graphs, and selects Triton attention plus PyTorch sampling for the smoke path.
Override `MODEL_PATH` or `PORT` only for local experiments.

Basic API checks after launch:

```bash
curl -s http://127.0.0.1:8011/v1/models | jq .
curl -s http://127.0.0.1:8011/model_info | jq .
```

The first real request may still trigger FlashInfer runtime cached-op generation;
after that happens, rerun `scripts/bigballs/prebuild_flashinfer_cached_ops.sh` to
make existing cache builds repeatable and parallel.

## Production guardrails

- Keep this lane clearly labeled as scratch/local for blackhouse.
- Do not modify production `llama.cpp`, `llmbrain.service`, systemd units, or
  production model paths.
- Do not hand-edit site-packages or FlashInfer cache files as the final state;
  use the repo-owned scripts above so changes are repeatable.
- Do not commit generated files: `~/.cache/flashinfer`, wheels under
  `sgl-kernel/dist`, build directories, downloaded models, venvs, or logs.
