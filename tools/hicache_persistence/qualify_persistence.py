#!/usr/bin/env python3
"""Black-box HiCache/NIXL persistence qualification for an OpenAI server.

The harness uses explicit lifecycle command arrays from a JSON config. It never
sends signals to request clients or the inference server. The full mode proves:

* fresh write-through into one representation namespace;
* service-restart storage reuse with exact long-context retrieval;
* concurrent restore;
* optional A -> B -> A namespace isolation and rollback reuse.

Two additional modes support a human-controlled full-machine reboot between
``reboot-seed`` and ``reboot-verify``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_STORAGE_HIT_METRIC = {
    "name": "sglang:prefill_effective_tokens_total",
    "labels": {"mode": "storage_hit"},
}
DEFAULT_BACKUP_METRIC = {
    "name": "sglang:backuped_tokens_total",
    "labels": {"storage_backend": "nixl"},
}


class QualificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    payload: dict[str, Any]
    needles: tuple[str, str, str]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualificationError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command(config: dict[str, Any], name: str, *, required: bool = True) -> list[str] | None:
    value = config.get("lifecycle", {}).get(name)
    if value is None and not required:
        return None
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise QualificationError(f"lifecycle.{name} must be a non-empty string array")
    return value


def run_command(config: dict[str, Any], name: str, *, required: bool = True) -> str:
    argv = command(config, name, required=required)
    if argv is None:
        return ""
    timeout = float(config.get("command_timeout_seconds", 900))
    result = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise QualificationError(
            f"lifecycle.{name} failed with exit {result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def url(config: dict[str, Any], path: str) -> str:
    return str(config["base_url"]).rstrip("/") + path


def get_text(config: dict[str, Any], path: str, timeout: float = 10) -> str:
    with urllib.request.urlopen(url(config, path), timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def wait_ready(config: dict[str, Any]) -> None:
    deadline = time.monotonic() + float(config.get("ready_timeout_seconds", 1200))
    expected_model = str(config["served_model"])
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            payload = json.loads(get_text(config, "/v1/models"))
            ids = {str(item.get("id")) for item in payload.get("data", [])}
            if expected_model in ids:
                return
            last_error = f"served models are {sorted(ids)}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise QualificationError(f"server did not become ready: {last_error}")


def probe_namespace(config: dict[str, Any]) -> Path:
    raw = run_command(config, "namespace_probe")
    if not raw:
        raise QualificationError("namespace_probe returned no path")
    # Permit a probe to log before printing the selected root; the last line owns it.
    root = Path(raw.splitlines()[-1]).expanduser().resolve()
    if not root.is_dir():
        raise QualificationError(f"selected namespace is not a directory: {root}")
    return root


def namespace_snapshot(root: Path) -> dict[str, Any]:
    entries = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        relative = str(path.relative_to(root))
        entries.append((relative, stat.st_size, stat.st_mtime_ns))
        total_bytes += stat.st_size
    encoded = json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode()
    return {
        "root": str(root),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "listing_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        key, raw_value = line.rsplit(" ", 1)
        try:
            values[key] = float(raw_value)
        except ValueError:
            continue
    return values


def metric_value(values: dict[str, float], spec: dict[str, Any]) -> float:
    name = str(spec["name"])
    labels = spec.get("labels", {})
    required = [f'{key}="{value}"' for key, value in sorted(labels.items())]
    matches = [
        value
        for key, value in values.items()
        if (key == name or key.startswith(name + "{"))
        and all(label in key for label in required)
    ]
    if not matches:
        raise QualificationError(
            f"required metric not found: {name} labels={labels}; "
            "adjust the metric spec for this SGLang revision"
        )
    return sum(matches)


def metrics_snapshot(config: dict[str, Any]) -> dict[str, float]:
    return parse_metrics(get_text(config, "/metrics", timeout=30))


def stream_request(config: dict[str, Any], case: PromptCase) -> dict[str, Any]:
    request = urllib.request.Request(
        url(config, "/v1/chat/completions"),
        data=json.dumps(case.payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    first_token = None
    usage = None
    finish_reason = None
    content: list[str] = []
    reasoning: list[str] = []
    timeout = float(config.get("request_timeout_seconds", 3600))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                visible = delta.get("content") or ""
                hidden = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if visible or hidden:
                    first_token = first_token or time.time()
                content.append(visible)
                reasoning.append(hidden)
    ended = time.time()
    if usage is None:
        raise QualificationError(f"{case.case_id}: streaming response had no usage")
    combined = ("".join(reasoning) + "\n" + "".join(content)).upper()
    missing = [needle for needle in case.needles if needle.upper() not in combined]
    if missing:
        raise QualificationError(f"{case.case_id}: missing exact needles: {missing}")
    return {
        "case_id": case.case_id,
        "started_epoch": started,
        "first_token_epoch": first_token,
        "ended_epoch": ended,
        "wall_seconds": ended - started,
        "ttft_seconds": first_token - started if first_token else None,
        "usage": usage,
        "finish_reason": finish_reason,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "needles": list(case.needles),
    }


def run_cases(config: dict[str, Any], cases: list[PromptCase]) -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(stream_request, config, case) for case in cases]
        return [future.result() for future in futures]


def build_cases(config: dict[str, Any]) -> list[PromptCase]:
    count = int(config.get("concurrency", 3))
    if count < 1:
        raise QualificationError("concurrency must be positive")
    target_chars = int(config.get("prompt_target_chars", 240000))
    if target_chars < 10000:
        raise QualificationError("prompt_target_chars must be at least 10000")
    marker = str(config.get("prompt_identity", "HICACHE-PERSISTENCE-V1"))
    model = str(config["served_model"])
    reasoning_effort = str(config.get("reasoning_effort", "medium"))
    cases = []
    for index in range(count):
        digest = hashlib.sha256(f"{marker}:{index}".encode()).hexdigest().upper()
        needles = (
            f"ALPHA-{digest[:12]}",
            f"BRAVO-{digest[12:24]}",
            f"CHARLIE-{digest[24:36]}",
        )
        prefix = (
            f"Archive {marker}-{index}. Read the complete archive. "
            "At the end, return all three exceptional codes exactly, in the form "
            "ALPHA=<code>; BRAVO=<code>; CHARLIE=<code>.\n"
        )
        filler = "Routine record: pumps, labels, access gates, and weather seals are normal.\n"
        filler_count = max(1, (target_chars - len(prefix) - 256) // len(filler))
        positions = {filler_count // 4: needles[0], filler_count // 2: needles[1], 3 * filler_count // 4: needles[2]}
        lines = [prefix]
        labels = ("ALPHA", "BRAVO", "CHARLIE")
        needle_index = 0
        for line_index in range(filler_count):
            if line_index in positions:
                lines.append(f"EXCEPTIONAL {labels[needle_index]}: {positions[line_index]}.\n")
                needle_index += 1
            lines.append(filler)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "".join(lines)}],
            "max_tokens": int(config.get("max_output_tokens", 512)),
            "temperature": 0,
            "seed": int(config.get("seed", 20260826)) + index,
            "reasoning_effort": reasoning_effort,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        cases.append(PromptCase(f"archive-{index}", payload, needles))
    return cases


def metric_specs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = config.get("metrics", {})
    return (
        metrics.get("storage_hit", DEFAULT_STORAGE_HIT_METRIC),
        metrics.get("backup", DEFAULT_BACKUP_METRIC),
    )


def wait_for_backup(
    config: dict[str, Any],
    before: dict[str, float],
    expected_tokens: int,
) -> dict[str, Any]:
    _, backup_spec = metric_specs(config)
    baseline = metric_value(before, backup_spec)
    minimum_ratio = float(config.get("minimum_backup_ratio", 0.90))
    target = expected_tokens * minimum_ratio
    deadline = time.monotonic() + float(config.get("backup_timeout_seconds", 300))
    latest = before
    delta = 0.0
    while time.monotonic() < deadline:
        latest = metrics_snapshot(config)
        delta = metric_value(latest, backup_spec) - baseline
        if delta >= target:
            return {"backup_tokens": delta, "required_tokens": target}
        time.sleep(1)
    raise QualificationError(
        f"write-through backup did not reach {target:.0f} tokens; observed {delta:.0f}"
    )


def storage_restore(
    config: dict[str, Any], cases: list[PromptCase]
) -> dict[str, Any]:
    storage_spec, _ = metric_specs(config)
    before = metrics_snapshot(config)
    before_value = metric_value(before, storage_spec)
    results = run_cases(config, cases)
    after = metrics_snapshot(config)
    restored = metric_value(after, storage_spec) - before_value
    prompt_tokens = sum(int(row["usage"]["prompt_tokens"]) for row in results)
    ratio = restored / prompt_tokens if prompt_tokens else 0.0
    return {
        "results": results,
        "storage_hit_tokens": restored,
        "prompt_tokens": prompt_tokens,
        "storage_hit_ratio": ratio,
    }


def assert_restore_ratio(config: dict[str, Any], result: dict[str, Any], label: str) -> None:
    minimum = float(config.get("minimum_restore_ratio", 0.90))
    if result["storage_hit_ratio"] < minimum:
        raise QualificationError(
            f"{label}: storage-hit ratio {result['storage_hit_ratio']:.4f} is below {minimum:.4f}"
        )


def transition(config: dict[str, Any], start_name: str) -> None:
    run_command(config, "stop")
    run_command(config, start_name)
    wait_ready(config)


def fresh_seed(config: dict[str, Any], cases: list[PromptCase]) -> tuple[Path, dict[str, Any]]:
    run_command(config, "stop")
    run_command(config, "clear_original")
    run_command(config, "start_original")
    wait_ready(config)
    namespace = probe_namespace(config)
    before = metrics_snapshot(config)
    results = run_cases(config, cases)
    prompt_tokens = sum(int(row["usage"]["prompt_tokens"]) for row in results)
    backup = wait_for_backup(config, before, prompt_tokens)
    return namespace, {
        "results": results,
        "prompt_tokens": prompt_tokens,
        "backup": backup,
        "namespace": namespace_snapshot(namespace),
    }


def run_full(config: dict[str, Any]) -> dict[str, Any]:
    cases = build_cases(config)
    namespace_a, seed = fresh_seed(config, cases)
    snapshot_a = namespace_snapshot(namespace_a)

    transition(config, "start_original")
    restarted_a = probe_namespace(config)
    if restarted_a != namespace_a:
        raise QualificationError(
            f"identical restart selected {restarted_a}, expected {namespace_a}"
        )
    restart_restore = storage_restore(config, cases)
    assert_restore_ratio(config, restart_restore, "identical restart")

    result: dict[str, Any] = {
        "seed": seed,
        "restart_restore": restart_restore,
        "namespace_a": snapshot_a,
    }

    variant_command = command(config, "start_variant", required=False)
    if variant_command is not None:
        transition(config, "start_variant")
        namespace_b = probe_namespace(config)
        if namespace_b == namespace_a:
            raise QualificationError("representation variant reused namespace A")
        variant = storage_restore(config, cases[:1])
        maximum_cold_ratio = float(config.get("maximum_variant_storage_hit_ratio", 0.10))
        if variant["storage_hit_ratio"] > maximum_cold_ratio:
            raise QualificationError(
                f"variant unexpectedly reused storage: ratio={variant['storage_hit_ratio']:.4f}"
            )
        after_variant_a = namespace_snapshot(namespace_a)
        if after_variant_a["listing_sha256"] != snapshot_a["listing_sha256"]:
            raise QualificationError("variant configuration modified namespace A")

        transition(config, "start_original")
        rollback_a = probe_namespace(config)
        if rollback_a != namespace_a:
            raise QualificationError(
                f"rollback selected {rollback_a}, expected original {namespace_a}"
            )
        rollback = storage_restore(config, cases[:1])
        assert_restore_ratio(config, rollback, "A -> B -> A rollback")
        result.update(
            {
                "namespace_b": namespace_snapshot(namespace_b),
                "variant_cold": variant,
                "namespace_a_after_variant": after_variant_a,
                "rollback_restore": rollback,
            }
        )
    elif bool(config.get("require_variant", True)):
        raise QualificationError(
            "require_variant is true but lifecycle.start_variant is not configured"
        )

    return result


def run_reboot_seed(config: dict[str, Any], state_file: Path) -> dict[str, Any]:
    cases = build_cases(config)
    namespace, seed = fresh_seed(config, cases)
    state = {
        "schema": "sglang-hicache-persistence-reboot-v1",
        "namespace": str(namespace),
        "namespace_snapshot": namespace_snapshot(namespace),
        "prompt_identity": config.get("prompt_identity", "HICACHE-PERSISTENCE-V1"),
        "concurrency": int(config.get("concurrency", 3)),
        "prompt_target_chars": int(config.get("prompt_target_chars", 240000)),
        "seed": seed,
    }
    write_json(state_file, state)
    return state


def run_reboot_verify(config: dict[str, Any], state_file: Path) -> dict[str, Any]:
    state = read_json(state_file)
    if state.get("schema") != "sglang-hicache-persistence-reboot-v1":
        raise QualificationError("unsupported reboot state schema")
    for key in ("prompt_identity", "concurrency", "prompt_target_chars"):
        expected = state[key]
        actual = config.get(key, "HICACHE-PERSISTENCE-V1" if key == "prompt_identity" else (3 if key == "concurrency" else 240000))
        if actual != expected:
            raise QualificationError(f"reboot config mismatch for {key}: {actual!r} != {expected!r}")
    wait_ready(config)
    namespace = probe_namespace(config)
    if str(namespace) != state["namespace"]:
        raise QualificationError(
            f"post-reboot namespace {namespace} != seeded namespace {state['namespace']}"
        )
    restore = storage_restore(config, build_cases(config))
    assert_restore_ratio(config, restore, "full-machine reboot")
    return {"namespace": namespace_snapshot(namespace), "reboot_restore": restore}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("full", "reboot-seed", "reboot-verify"),
        default="full",
    )
    parser.add_argument("--state-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = read_json(args.config)
    for required in ("base_url", "served_model"):
        if not config.get(required):
            raise QualificationError(f"missing required config field: {required}")
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "mode": args.mode,
        "started_epoch": time.time(),
        "status": "running",
    }
    original_may_need_restore = args.mode == "full"
    try:
        if args.mode == "full":
            report["result"] = run_full(config)
        elif args.mode == "reboot-seed":
            if args.state_file is None:
                raise QualificationError("--state-file is required for reboot-seed")
            report["result"] = run_reboot_seed(config, args.state_file)
        else:
            if args.state_file is None:
                raise QualificationError("--state-file is required for reboot-verify")
            report["result"] = run_reboot_verify(config, args.state_file)
        report["status"] = "passed"
        return_code = 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return_code = 1
    finally:
        if original_may_need_restore:
            try:
                transition(config, "start_original")
                report["final_namespace"] = str(probe_namespace(config))
                report["original_runtime_restored"] = True
            except Exception as exc:
                report["original_runtime_restored"] = False
                report["restore_error"] = f"{type(exc).__name__}: {exc}"
                return_code = 1
        report["ended_epoch"] = time.time()
        write_json(args.output / "qualification.json", report)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
