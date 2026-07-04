#!/usr/bin/env bash
# Scratch-only patch for /opt/bigballs SGLang experiments on RTX PRO 6000 Blackwell.
# It reduces sgl-kernel CUDA codegen to the SM120a path, disables the duplicate
# common_ops_sm90 target, disables FlashMLA for the first probe, and keeps nvcc
# compile threading sane.  This is not intended as an upstream-clean patch.
set -euo pipefail

if [[ ! -f CMakeLists.txt || ! -d cmake ]]; then
  echo "error: run this from sgl-kernel/" >&2
  exit 1
fi

cp -n CMakeLists.txt CMakeLists.txt.bigballs.bak || true
cp -n cmake/flashmla.cmake cmake/flashmla.cmake.bigballs.bak || true

python - <<'PY'
from pathlib import Path


def replace_once(s: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if old in s:
        return s.replace(old, new, 1), True
    if new in s:
        return s, False
    print(f"warning: pattern not found, already patched or upstream changed: {label}")
    return s, False

p = Path("CMakeLists.txt")
s = p.read_text()
changed = False

repls = [
    (
        '    "-gencode=arch=compute_90,code=sm_90"\n',
        '    # bigballs: SM120-only scratch build; drop baseline sm_90 codegen.\n',
        'drop baseline sm_90 gencode',
    ),
    (
        'set(SGL_KERNEL_COMPILE_THREADS 32 CACHE STRING "Set compilation threads, default 32")',
        'set(SGL_KERNEL_COMPILE_THREADS 1 CACHE STRING "Set compilation threads, default 1")',
        'default compile threads 32->1',
    ),
    (
        '        "-gencode=arch=compute_100a,code=sm_100a"\n        "-gencode=arch=compute_120a,code=sm_120a"',
        '        # bigballs: skip sm_100a for first RTX PRO 6000 SM120 probe.\n        "-gencode=arch=compute_120a,code=sm_120a"',
        'drop sm_100a gencode',
    ),
    (
        '            "-gencode=arch=compute_103a,code=sm_103a"\n            "--compress-mode=size"',
        '            # bigballs: skip sm_103a for first RTX PRO 6000 SM120 probe.\n            "--compress-mode=size"',
        'drop sm_103a gencode',
    ),
    (
        'if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" AND SGL_KERNEL_ENABLE_FA3)\n    list(APPEND SGL_KERNEL_CUDA_FLAGS\n        "-gencode=arch=compute_90a,code=sm_90a"\n    )\nendif()',
        '# bigballs: disable FA3 sm_90a codegen for first SM120-only scratch build.\n#if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" AND SGL_KERNEL_ENABLE_FA3)\n#    list(APPEND SGL_KERNEL_CUDA_FLAGS\n#        "-gencode=arch=compute_90a,code=sm_90a"\n#    )\n#endif()',
        'disable FA3 sm_90a gencode block',
    ),
    (
        'include(${CMAKE_CURRENT_LIST_DIR}/cmake/flashmla.cmake)',
        '# bigballs: disable FlashMLA for first SM120-only scratch build; it injects sm90/sm100/sm103 paths.\n#include(${CMAKE_CURRENT_LIST_DIR}/cmake/flashmla.cmake)',
        'disable FlashMLA include',
    ),
]

for old, new, label in repls:
    s, did = replace_once(s, old, new, label)
    changed = changed or did

sm90_block_old = '''# =========================== Common SM90 Build ============================= #
# Build SM90 library with fast math optimization (same namespace, different directory)
Python_add_library(common_ops_sm90_build MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

target_compile_options(common_ops_sm90_build PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${SGL_KERNEL_CUDA_FLAGS} -use_fast_math>
)
target_include_directories(common_ops_sm90_build PRIVATE ${INCLUDES})
# Set output name and separate build directory to avoid conflicts
set_target_properties(common_ops_sm90_build PROPERTIES
    OUTPUT_NAME "common_ops"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sm90"
)
'''
sm90_block_new = '''# =========================== Common SM90 Build ============================= #
# bigballs: disabled duplicate common_ops_sm90_build for first SM120-only probe.
# The sm100 target below is still compiled with SM120a codegen and installed under sgl_kernel/sm100.
'''
s, did = replace_once(s, sm90_block_old, sm90_block_new, 'disable common_ops_sm90 target block')
changed = changed or did

link_old = 'target_link_libraries(common_ops_sm90_build PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)\n'
link_new = '# bigballs: common_ops_sm90_build disabled.\n# target_link_libraries(common_ops_sm90_build PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)\n'
s, did = replace_once(s, link_old, link_new, 'disable common_ops_sm90 link')
changed = changed or did

defs_old = '''target_compile_definitions(common_ops_sm90_build PRIVATE
    FLASHATTENTION_DISABLE_BACKWARD
    FLASHATTENTION_DISABLE_DROPOUT
    FLASHATTENTION_DISABLE_UNEVEN_K
)
'''
defs_new = '''# bigballs: common_ops_sm90_build disabled.
# target_compile_definitions(common_ops_sm90_build PRIVATE
#     FLASHATTENTION_DISABLE_BACKWARD
#     FLASHATTENTION_DISABLE_DROPOUT
#     FLASHATTENTION_DISABLE_UNEVEN_K
# )
'''
s, did = replace_once(s, defs_old, defs_new, 'disable common_ops_sm90 compile definitions')
changed = changed or did

install_old = 'install(TARGETS common_ops_sm90_build LIBRARY DESTINATION sgl_kernel/sm90)\n'
install_new = '# bigballs: common_ops_sm90_build disabled.\n# install(TARGETS common_ops_sm90_build LIBRARY DESTINATION sgl_kernel/sm90)\n'
s, did = replace_once(s, install_old, install_new, 'disable common_ops_sm90 install')
changed = changed or did

p.write_text(s)
print("patched CMakeLists.txt" if changed else "CMakeLists.txt already looked patched")

p = Path("cmake/flashmla.cmake")
s = p.read_text()
changed = False
repls = [
    (
        '    list(APPEND FLASHMLA_CUDA_FLAGS\n        "-gencode=arch=compute_90a,code=sm_90a"\n    )',
        '    # bigballs: disabled sm_90a codegen for first SM120-only scratch build.\n    # list(APPEND FLASHMLA_CUDA_FLAGS\n    #     "-gencode=arch=compute_90a,code=sm_90a"\n    # )',
        'flashmla drop sm_90a',
    ),
    (
        '    list(APPEND FLASHMLA_CUDA_FLAGS\n        "-gencode=arch=compute_100a,code=sm_100a"\n    )\n    set(FLASHMLA_ENABLE_SM100 ON)',
        '    # bigballs: disabled sm_100a codegen and SM100 source expansion for first SM120-only scratch build.\n    # list(APPEND FLASHMLA_CUDA_FLAGS\n    #     "-gencode=arch=compute_100a,code=sm_100a"\n    # )\n    set(FLASHMLA_ENABLE_SM100 OFF)',
        'flashmla drop sm_100a',
    ),
    (
        '    list(APPEND FLASHMLA_CUDA_FLAGS\n        "-gencode=arch=compute_103a,code=sm_103a"\n    )',
        '    # bigballs: disabled sm_103a codegen for first SM120-only scratch build.\n    # list(APPEND FLASHMLA_CUDA_FLAGS\n    #     "-gencode=arch=compute_103a,code=sm_103a"\n    # )',
        'flashmla drop sm_103a',
    ),
]
for old, new, label in repls:
    s, did = replace_once(s, old, new, label)
    changed = changed or did
p.write_text(s)
print("patched cmake/flashmla.cmake" if changed else "cmake/flashmla.cmake already looked patched")
PY

echo "Remaining gencode lines in edited files:"
grep -RIn -- '-gencode=arch=compute_' CMakeLists.txt cmake/flashmla.cmake || true

echo
echo "Remaining common_ops_sm90_build references:"
grep -RIn -- 'common_ops_sm90_build' CMakeLists.txt || true
