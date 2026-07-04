#!/usr/bin/env python3
"""Patch installed FlashInfer runtime cached-op generators for blackhouse builds.

Scratch/local helper: edits the installed FlashInfer Python package so future
cached-op build.ninja files and runtime Ninja invocations use concrete values
read from FLASHINFER_NINJA_JOBS and FLASHINFER_NVCC_THREADS.

The real runtime generator in modern FlashInfer lives under flashinfer/jit
(usually flashinfer/jit/cpp_ext.py).  Vendored examples under flashinfer/data are
ignored by default because patching them does not affect ~/.cache/flashinfer
cached-op builds.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

MARKER = "bigballs-flashinfer-parallelism"
DEFAULT_NINJA_JOBS = "24"
DEFAULT_NVCC_THREADS = "4"
EXCLUDED_PART_SEQUENCES = (
    ("data", "cccl", "ci"),
    ("data", "cutlass", "examples"),
)
REAL_GENERATOR_HINTS = (
    "generate_ninja_build_for_op",
    "get_nvcc_parallelism_flags",
    "run_ninja",
    "FLASHINFER_JIT_DIR",
    "cached_ops",
)


@dataclass(frozen=True)
class PatchResult:
    path: Path
    replacements: int
    excluded: bool
    real_generator: bool


def flashinfer_root() -> Path:
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or spec.origin is None:
        raise SystemExit("error: flashinfer is not importable in this Python environment")
    return Path(spec.origin).resolve().parent


def has_part_sequence(path: Path, sequence: tuple[str, ...]) -> bool:
    parts = path.parts
    width = len(sequence)
    return any(tuple(parts[i : i + width]) == sequence for i in range(len(parts) - width + 1))


def is_excluded_path(path: Path) -> bool:
    return any(has_part_sequence(path, sequence) for sequence in EXCLUDED_PART_SEQUENCES)


def looks_like_real_generator(path: Path, text: str) -> bool:
    parts = set(path.parts)
    if "jit" in parts and ("build.ninja" in text or "ninja" in text and "nvcc" in text):
        return True
    return any(hint in text for hint in REAL_GENERATOR_HINTS)


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
    new_value, count = re.subn(r"--threads([= ]+)[0-9]+", rf"--threads\1{thread_placeholder}", value)
    new_value, job_count = re.subn(r"(?<![\w-])-j ?[0-9]+(?!\d)", rf"-j{jobs_placeholder}", new_value)
    count += job_count
    if count == 0:
        return token_text, 0

    escaped = new_value.replace("{", "{{").replace("}", "}}")
    escaped = escaped.replace(thread_placeholder, "{os.environ.get('FLASHINFER_NVCC_THREADS', '4')}").replace(
        jobs_placeholder, "{os.environ.get('FLASHINFER_NINJA_JOBS', '24')}"
    )

    if "\n" in escaped:
        escaped = escaped.replace("\\", "\\\\").replace('\"\"\"', '\\\"\\\"\\\"')
        return 'f"""' + escaped + '"""', count
    escaped = escaped.replace("\\", "\\\\").replace('"', '\\"')
    return 'f"' + escaped + '"', count


def patch_text(text: str) -> tuple[str, int]:
    original = text
    count = 0

    # Modern FlashInfer runtime JIT uses helpers in flashinfer/jit/cpp_ext.py.
    text, n = re.subn(
        r'(env_var_name\s*=\s*["\']FLASHINFER_NVCC_THREADS["\']\s*\n\s*default\s*=\s*)1\b',
        rf"\g<1>{DEFAULT_NVCC_THREADS}",
        text,
    )
    count += n
    text, n = re.subn(
        r"def _get_num_workers\(\) -> Optional\[int\]:\n"
        r"    max_jobs = os\.environ\.get\(\"MAX_JOBS\"\)\n"
        r"    if max_jobs is not None and max_jobs\.isdigit\(\):\n"
        r"        return int\(max_jobs\)\n"
        r"    return None",
        "def _get_num_workers() -> Optional[int]:\n"
        "    for env_var_name, default in ((\"FLASHINFER_NINJA_JOBS\", \"24\"), (\"MAX_JOBS\", None)):\n"
        "        value = os.environ.get(env_var_name, default)\n"
        "        if value is not None and value.isdigit():\n"
        "            return int(value)\n"
        "    return None",
        text,
    )
    count += n

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
    parser.add_argument("--dry-run", action="store_true", help="report files that would be patched without writing them")
    parser.add_argument("--verbose", action="store_true", help="report candidate files, including skipped vendored/example files")
    parser.add_argument("--include-vendored-examples", action="store_true", help="also patch flashinfer/data vendored CI/example trees")
    args = parser.parse_args()

    os.environ.setdefault("FLASHINFER_NINJA_JOBS", DEFAULT_NINJA_JOBS)
    os.environ.setdefault("FLASHINFER_NVCC_THREADS", DEFAULT_NVCC_THREADS)

    root = (args.package_root or flashinfer_root()).resolve()
    if not root.is_dir():
        raise SystemExit(f"error: not a directory: {root}")

    results: list[PatchResult] = []
    skipped_excluded: list[Path] = []
    total = 0
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        if not any(needle in text for needle in ("nvcc", "--threads", "-j", "MAX_JOBS", "FLASHINFER_NVCC_THREADS")):
            continue
        excluded = is_excluded_path(path)
        if excluded and not args.include_vendored_examples:
            if args.verbose:
                skipped_excluded.append(path)
            continue
        new_text, count = patch_text(text)
        if count:
            real_generator = looks_like_real_generator(path, text)
            total += count
            results.append(PatchResult(path, count, excluded, real_generator))
            if not args.dry_run:
                backup = path.with_suffix(path.suffix + ".bigballs.bak")
                if not backup.exists():
                    backup.write_text(text)
                path.write_text(new_text)

    real_results = [result for result in results if result.real_generator and not result.excluded]
    action = "Would patch" if args.dry_run else "Patched"
    if results:
        print(f"{action} {len(results)} file(s), {total} replacement(s), under {root}")
        for result in results:
            tags = []
            if result.real_generator:
                tags.append("runtime-generator")
            if result.excluded:
                tags.append("vendored/example")
            print(f"{result.path} ({result.replacements} replacement(s){'; ' + ', '.join(tags) if tags else ''})")
    else:
        print(f"No hard-coded FlashInfer nvcc/ninja thread defaults found under {root}")

    if skipped_excluded:
        print("Skipped vendored/example candidate file(s):")
        for path in skipped_excluded:
            print(path)

    if not real_results:
        print(
            "error: no real FlashInfer runtime cached-op generator was patched. "
            "Vendored/example files do not affect ~/.cache/flashinfer cached_ops builds.",
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
