#!/usr/bin/env bash
# Scratch/local blackhouse helper: install and sanity-check local sglang-kernel.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WHEEL="${1:-}"
if [[ -z "$WHEEL" ]]; then
  WHEEL="$(find "$REPO_ROOT/sgl-kernel/dist" -maxdepth 1 -type f -name '*.whl' | sort | tail -n1 || true)"
fi
if [[ -z "$WHEEL" || ! -f "$WHEEL" ]]; then echo "error: wheel not found; pass path or build first" >&2; exit 1; fi
"$PYTHON_BIN" -m pip install --force-reinstall --no-deps "$WHEEL"
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import sgl_kernel
print("sgl_kernel import:", sgl_kernel.__file__)
try:
    print("sglang-kernel version:", md.version("sglang-kernel"))
except md.PackageNotFoundError:
    print("sglang-kernel distribution metadata not found")
PY
