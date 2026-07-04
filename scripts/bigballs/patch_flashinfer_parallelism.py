#!/usr/bin/env python3
"""Patch installed FlashInfer cached-op generators for blackhouse local builds.

Scratch/local helper: edits the installed FlashInfer Python package so future
cached-op build.ninja files use FLASHINFER_NINJA_JOBS and
FLASHINFER_NVCC_THREADS instead of hard-coded single-thread defaults.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

MARKER = "bigballs-flashinfer-parallelism"
DEFAULT_NINJA_JOBS = "24"
DEFAULT_NVCC_THREADS = "4"


def flashinfer_root() -> Path:
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or spec.origin is None:
        raise SystemExit("error: flashinfer is not importable in this Python environment")
    return Path(spec.origin).resolve().parent


def patch_text(text: str) -> tuple[str, int]:
    original = text
    count = 0

    replacements = [
        (r"--threads=1", "--threads=${FLASHINFER_NVCC_THREADS}"),
        (r"--threads 1", "--threads ${FLASHINFER_NVCC_THREADS}"),
        (r"--threads\", \"1\"", "--threads\", os.environ.get(\"FLASHINFER_NVCC_THREADS\", \"4\")"),
        (r"--threads', '1'", "--threads', os.environ.get('FLASHINFER_NVCC_THREADS', '4')"),
    ]
    for old, new in replacements:
        n = text.count(old)
        if n:
            text = text.replace(old, new)
            count += n

    # Handle common generated-Ninja job-pool defaults without changing normal ninja CLI use.
    for pattern in [r"(?<![\w-])-j1(?!\d)", r"(?<![\w-])-j 1(?!\d)"]:
        text, n = re.subn(pattern, "-j${FLASHINFER_NINJA_JOBS}", text)
        count += n

    if count and "FLASHINFER_NVCC_THREADS" in text and "import os" not in text:
        lines = text.splitlines()
        insert_at = 0
        while insert_at < len(lines) and (lines[insert_at].startswith("#!") or lines[insert_at].startswith("#")):
            insert_at += 1
        if insert_at < len(lines) and lines[insert_at].startswith(('"""', "'''")):
            quote = lines[insert_at][:3]
            insert_at += 1
            while insert_at < len(lines) and quote not in lines[insert_at]:
                insert_at += 1
            insert_at += 1
        while insert_at < len(lines) and lines[insert_at].startswith("from __future__"):
            insert_at += 1
        lines.insert(insert_at, "import os  # " + MARKER)
        text = "\n".join(lines) + ("\n" if original.endswith("\n") else "")

    return text, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=None, help="FlashInfer package directory; defaults to import location")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("FLASHINFER_NINJA_JOBS", DEFAULT_NINJA_JOBS)
    os.environ.setdefault("FLASHINFER_NVCC_THREADS", DEFAULT_NVCC_THREADS)

    root = (args.package_root or flashinfer_root()).resolve()
    if not root.is_dir():
        raise SystemExit(f"error: not a directory: {root}")

    touched: list[Path] = []
    total = 0
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        if "nvcc" not in text and "--threads" not in text and "-j1" not in text:
            continue
        new_text, count = patch_text(text)
        if count:
            total += count
            touched.append(path)
            if not args.dry_run:
                backup = path.with_suffix(path.suffix + ".bigballs.bak")
                if not backup.exists():
                    backup.write_text(text)
                path.write_text(new_text)

    if not touched:
        print(f"No hard-coded FlashInfer nvcc/ninja single-thread defaults found under {root}")
        return 0
    action = "Would patch" if args.dry_run else "Patched"
    print(f"{action} {len(touched)} file(s), {total} replacement(s), under {root}")
    for path in touched:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
