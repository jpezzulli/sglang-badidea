#!/usr/bin/env bash
# Scratch-only patch for /opt/bigballs SGLang experiments on RTX PRO 6000 Blackwell.
# It reduces sgl-kernel CUDA codegen to the SM120a path and keeps nvcc compile
# threading sane.  This is not intended as an upstream-clean patch.
set -euo pipefail

if [[ ! -f CMakeLists.txt || ! -d cmake ]]; then
  echo "error: run this from sgl-kernel/" >&2
  exit 1
fi

cp -n CMakeLists.txt CMakeLists.txt.bigballs.bak || true
cp -n cmake/flashmla.cmake cmake/flashmla.cmake.bigballs.bak || true

python - <<'PY'
from pathlib import Path

p = Path("CMakeLists.txt")
s = p.read_text()

repls = [
    (
        '    "-gencode=arch=compute_90,code=sm_90"\n',
        '    # bigballs: SM120-only scratch build; drop baseline sm_90 codegen.\n'
    ),
    (
        'set(SGL_KERNEL_COMPILE_THREADS 32 CACHE STRING "Set compilation threads, default 32")',
        'set(SGL_KERNEL_COMPILE_THREADS 1 CACHE STRING "Set compilation threads, default 1")'
    ),
    (
        '        "-gencode=arch=compute_100a,code=sm_100a"\n        "-gencode=arch=compute_120a,code=sm_120a"',
        '        # bigballs: skip sm_100a for first RTX PRO 6000 SM120 probe.\n        "-gencode=arch=compute_120a,code=sm_120a"'
    ),
    (
        '            "-gencode=arch=compute_103a,code=sm_103a"\n            "--compress-mode=size"',
        '            # bigballs: skip sm_103a for first RTX PRO 6000 SM120 probe.\n            "--compress-mode=size"'
    ),
    (
        'if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" AND SGL_KERNEL_ENABLE_FA3)\n    list(APPEND SGL_KERNEL_CUDA_FLAGS\n        "-gencode=arch=compute_90a,code=sm_90a"\n    )\nendif()',
        '# bigballs: disable FA3 sm_90a codegen for first SM120-only scratch build.\n#if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" AND SGL_KERNEL_ENABLE_FA3)\n#    list(APPEND SGL_KERNEL_CUDA_FLAGS\n#        "-gencode=arch=compute_90a,code=sm_90a"\n#    )\n#endif()'
    ),
    (
        'include(${CMAKE_CURRENT_LIST_DIR}/cmake/flashmla.cmake)',
        '# bigballs: disable FlashMLA for first SM120-only scratch build; it injects sm90/sm100/sm103 paths.\n#include(${CMAKE_CURRENT_LIST_DIR}/cmake/flashmla.cmake)'
    ),
]

changed = False
for old, new in repls:
    if old in s:
        s = s.replace(old, new, 1)
        changed = True
    elif new in s:
        pass
    else:
        print(f"warning: pattern not found, already patched or upstream changed: {old[:80]!r}")

p.write_text(s)
print("patched CMakeLists.txt" if changed else "CMakeLists.txt already looked patched")

p = Path("cmake/flashmla.cmake")
s = p.read_text()
repls = [
    (
        '    list(APPEND FLASHMLA_CUDA_FLAGS\n        "-gencode=arch=compute_90a,code=sm_90a"\n    )',
        '    # bigballs: disabled sm_90a codegen for first SM120-only scratch build.\n    # list(APPEND FLASHMLA_CUDA_FLAGS\n    #     "-gencode=arch=compute_90a,code=sm_90a"\n    # )'
    ),
    (
        '    list(APPEND FLASHMLA_CUDA_FLAGS\n        "-gencode=arch=compute_100a,code=sm_100a"\n    )\n    set(FLASHMLA_ENABLE_SM100 ON)',
        '    # bigballs: disabled sm_100a codegen and SM100 source expansion for first SM120-only scratch build.\n    # list(APPEND FLASHMLA_CUDA_FLAGS\n    #     "-gencode=arch=compute_100a,code=sm_100a"\n    # )\n    set(FLASHMLA_ENABLE_SM100 OFF)'
    ),
    (
        '    list(APPEND FLASHMLA_CUDA_FLAGS\n        "-gencode=arch=compute_103a,code=sm_103a"\n    )',
        '    # bigballs: disabled sm_103a codegen for first SM120-only scratch build.\n    # list(APPEND FLASHMLA_CUDA_FLAGS\n    #     "-gencode=arch=compute_103a,code=sm_103a"\n    # )'
    ),
]
changed = False
for old, new in repls:
    if old in s:
        s = s.replace(old, new, 1)
        changed = True
    elif new in s:
        pass
    else:
        print(f"warning: flashmla pattern not found, already patched or upstream changed: {old[:80]!r}")
p.write_text(s)
print("patched cmake/flashmla.cmake" if changed else "cmake/flashmla.cmake already looked patched")
PY

echo "Remaining gencode lines in edited files:"
grep -RIn -- '-gencode=arch=compute_' CMakeLists.txt cmake/flashmla.cmake || true
