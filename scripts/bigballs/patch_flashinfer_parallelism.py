#!/usr/bin/env python3
"""Patch installed FlashInfer cached-op generators for blackhouse local builds.

Scratch/local helper: edits the installed FlashInfer Python package so future
cached-op build.ninja files use concrete values read from FLASHINFER_NINJA_JOBS
and FLASHINFER_NVCC_THREADS instead of hard-coded single-thread defaults.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import re
import tokenize
from pathlib import Path

MARKER = "bigballs-flashinfer-parallelism"
DEFAULT_NINJA_JOBS = "24"
DEFAULT_NVCC_THREADS = "4"


def flashinfer_root() -> Path:
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or spec.origin is None:
        raise SystemExit("error: flashinfer is not importable in this Python environment")
    return Path(spec.origin).resolve().parent


def _patch_string_literal(token_text: str) -> tuple[str, int]:
    """Return a Python string token that emits concrete env values at runtime."""
    try:
        value = eval(token_text, {"__builtins__": {}})
    except Exception:
        return token_text, 0
    if not isinstance(value, str):
        return token_text, 0

    thread_placeholder = "__BIGBALLS_FLASHINFER_NVCC_THREADS__"
    jobs_placeholder = "__BIGBALLS_FLASHINFER_NINJA_JOBS__"
    new_value, count = re.subn(
        r"--threads([= ]+)[0-9]+",
        rf"--threads\1{thread_placeholder}",
        value,
    )
    new_value, job_count = re.subn(
        r"(?<![\w-])-j ?[0-9]+(?!\d)",
        rf"-j{jobs_placeholder}",
        new_value,
    )
    count += job_count
    if count == 0:
        return token_text, 0

    # Preserve existing braces in the generator string when converting it to an
    # f-string, then unescape only the env lookups inserted above.
    escaped = new_value.replace("{", "{{").replace("}", "}}")
    escaped = escaped.replace(
        thread_placeholder,
        "{os.environ.get('FLASHINFER_NVCC_THREADS', '4')}",
    ).replace(
        jobs_placeholder,
        "{os.environ.get('FLASHINFER_NINJA_JOBS', '24')}",
    )

    if "\n" in escaped:
        escaped = escaped.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        return 'f"""' + escaped + '"""', count
    escaped = escaped.replace("\\", "\\\\").replace('"', '\\"')
    return 'f"' + escaped + '"', count

def patch_text(text: str) -> tuple[str, int]:
    original = text
    count = 0

    # Patch standalone argument-list forms such as ["--threads", "1"] without
    # leaving raw Ninja variable syntax in generated build files.
    text, n = re.subn(
        r"([\"\']--threads[\"\']\s*,\s*)[\"\'][0-9]+[\"\']",
        r"\1os.environ.get('FLASHINFER_NVCC_THREADS', '4')",
        text,
    )
    count += n

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except tokenize.TokenError:
        return text, count

    patched_tokens = []
    for tok in tokens:
        if tok.type == tokenize.STRING:
            new_token, n = _patch_string_literal(tok.string)
            if n:
                tok = tokenize.TokenInfo(tok.type, new_token, tok.start, tok.end, tok.line)
                count += n
        patched_tokens.append(tok)
    text = tokenize.untokenize(patched_tokens)

    if count and "os.environ.get" in text and "import os" not in text:
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
        if "nvcc" not in text and "--threads" not in text and "-j" not in text:
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
        print(f"No hard-coded FlashInfer nvcc/ninja thread defaults found under {root}")
        return 0
    action = "Would patch" if args.dry_run else "Patched"
    print(f"{action} {len(touched)} file(s), {total} replacement(s), under {root}")
    for path in touched:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
