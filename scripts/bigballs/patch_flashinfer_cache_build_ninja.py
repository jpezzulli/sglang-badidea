#!/usr/bin/env python3
"""Patch already-generated FlashInfer cached build.ninja files."""
from __future__ import annotations
import argparse, os, re
from pathlib import Path


def patch(path: Path, nvcc_threads: str) -> bool:
    s = path.read_text()
    old = s
    s = re.sub(r"--threads([= ]+)[0-9]+", lambda m: f"--threads{m.group(1)}{nvcc_threads}", s)
    s = re.sub(r"(?<![\w-])-j1(?!\d)", f"-j{os.environ.get('FLASHINFER_NINJA_JOBS','24')}", s)
    s = re.sub(r"(?<![\w-])-j 1(?!\d)", f"-j {os.environ.get('FLASHINFER_NINJA_JOBS','24')}", s)
    if s != old:
        bak = path.with_suffix(path.suffix + ".bigballs.bak")
        if not bak.exists(): bak.write_text(old)
        path.write_text(s)
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, default=Path.home()/".cache"/"flashinfer")
    ap.add_argument("--nvcc-threads", default=os.environ.get("FLASHINFER_NVCC_THREADS", "4"))
    args = ap.parse_args()
    files = sorted(args.cache_root.glob("**/build.ninja"))
    changed = [p for p in files if patch(p, args.nvcc_threads)]
    print(f"Patched {len(changed)} of {len(files)} FlashInfer build.ninja file(s) under {args.cache_root}")
    for p in changed: print(p)
    return 0
if __name__ == "__main__": raise SystemExit(main())
