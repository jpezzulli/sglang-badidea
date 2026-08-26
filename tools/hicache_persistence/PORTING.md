# Porting Pennyroyal HiCache/NIXL persistence

This directory is the portable persistence layer extracted from the frozen Pennyroyal SGLang runtime. It is intentionally smaller than the runtime fork.

## Ordered patch series

Apply the generated git-format patches in order to the future SGLang `main`:

1. `fix(hicache): bound NIXL hybrid bounce transfers`
2. `fix(hicache): isolate concurrent NIXL path registrations`
3. `tools(hicache): add persistence porting kit`

Before applying patches 1 or 2, inspect current upstream for equivalent behavior and tests. Remove a patch when upstream has both the invariant and focused regression coverage; do not keep it merely because its commit hash differs.

Patch 3 is deployment tooling. It may remain on the local portability branch even if SGLang does not want launcher identity policy in core.

## Original failures

### Oversized bounce-backed hybrid transfer

`HiCacheNixl` pre-registers a fixed `STORAGE_BATCH_SIZE` bounce buffer for pools that cannot expose O_DIRECT-compatible zero-copy storage. The DFlash2 sidecar used this path. The v2 API accepted more logical pages than the bounce buffer contained and built descriptors/copy-back work as though every page had a slot.

The failure appeared only with sufficiently long prefixes. Short smoke tests and zero-copy target/Mamba pools did not exercise it. The correction chunks by logical page and slices `host_indices` by `page_size`, preserving alignment among logical keys, physical components, staging slots, page offsets, and returned results.

### Concurrent path registration identity collision

HiCache backup and prefetch run on separate threads. NIXL path mode keys active FILE registrations by `devId` and current NIXL explicitly requires each active path registration to use a unique value. Upstream SGLang generated `1..N` again for every FILE registration. Overlapping read/write registrations could therefore collide even when they named different files.

The correction uses one locked monotonic device-ID allocator for both FILE and OBJ registration. IDs need only be unique while registrations overlap; monotonic process-lifetime allocation is simpler and safe.

## Hybrid completeness invariant

For a hybrid runtime, target KV is only one part of a reusable prefix. The runtime must persist and restore every required state pool for the same page boundary:

```text
target KV
  + Mamba/GDN temporal state
  + every Mamba/GDN convolution component
  + packed or sidecar draft state when enabled
  = reusable hybrid prefix
```

Current upstream SGLang owns this controller and keying behavior. Do not recreate it in a local patch. The regression harness proves the combined behavior by requiring high measured L3 reuse and exact long-context output after process restart.

## Representation namespace invariant

NIXL FILE keys do not encode every tensor-layout assumption. Select a storage root from a canonical representation identity before launching SGLang:

```bash
export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR="$derived_root"
```

`derive_namespace.py` uses sorted canonical JSON, SHA-256, and a full manifest comparison. The readable directory suffix uses only the first 12 hex characters, but a truncated-hash collision cannot silently reuse data because the complete identity is checked in `namespace-identity.json`.

### Include when representation-sensitive

- target and draft checkpoint content identity, not only their paths;
- SGLang commit plus tracked diff;
- admitted context and RoPE/model overrides;
- TP, PP, attention-CP, DCP, and other topology that changes per-rank pool geometry;
- page size and memory layout;
- target and draft KV dtype and any non-unit KV scales;
- Mamba/GDN SSM and convolution dtypes;
- speculative algorithm, draft checkpoint, draft quantization, draft shape/window, and draft overrides;
- attention/prefill choices when they select a different serialized layout;
- Torch/CUDA ABI-relevant identity and accelerator architecture.

### Normally exclude

- request concurrency;
- `max_mamba_cache_size` and `mamba_max_states_per_path` while they only change allocator headroom;
- HiCache host capacity;
- cleaner thresholds;
- write/prefetch policy;
- power profile, logging, and sampling settings.

Being conservative is acceptable: an unnecessary new root loses reuse but does not reinterpret incompatible bytes.

## Checkpoint identity

For Hugging Face local-directory downloads, the helper uses the recorded immutable Hub revision and LFS SHA-256 for each shard. It also hashes model configuration/index files, safetensors headers, size, and mtime. When authoritative download metadata is absent, it hashes the entire weight payload.

Resolved paths are included deliberately. Relocating byte-identical weights therefore selects a new root. That is conservative and avoids assuming that two deployment paths have identical surrounding tokenizer/config assets.

## Root ownership

Current SGLang routes FILE existence queries, reads, writes, clear, and L3 cleaner scanning through `NixlFileManager` beneath the selected root. NIXL registrations and SGLang radix metadata remain process-local.

The active cleaner scans only the selected root. Inactive representation roots are isolated, but they require a separate operator lifecycle policy or their disk use can accumulate.

Cleaner high/low watermarks are percentages of the containing filesystem. They are not a namespace byte quota. Re-evaluate them when the cache moves to another filesystem or shares space with different data.

## Applying to a future runtime

1. Fetch the newest SGLang and NIXL `main`.
2. Record both exact commits; do not permanently pin them merely because they were tested.
3. Search current SGLang for the two patch invariants and focused tests.
4. Apply only the patches still missing.
5. Build NIXL and SGLang as one Python/CUDA/Torch/compiler compatibility unit.
6. Derive a fresh representation namespace with all applicable fields.
7. Start with write-through and an empty selected namespace.
8. Run focused unit tests.
9. Run `qualify_persistence.py --mode full` using explicit clean lifecycle wrappers.
10. Optionally run `reboot-seed`, reboot normally, then `reboot-verify` with the saved state file.

The lifecycle wrappers own systemd, launchers, and exact cache deletion. The regression script never sends Unix signals or PTY control characters and never invents a deletion target.

## Focused tests

From an environment that imports the worktree's SGLang source and the intended NIXL build:

```bash
python -m pytest -q \
  test/registered/unit/mem_cache/test_hicache_nixl_storage.py

python -m unittest -v \
  tools/hicache_persistence/test_derive_namespace.py \
  tools/hicache_persistence/test_qualify_persistence.py
```

The NIXL suite requires the NIXL Python bindings and a usable file backend. Some backend-dependent tests can skip when the installed NIXL lacks path mode.

## Full regression

Copy `example-config.json` and replace every lifecycle wrapper. Commands are JSON arrays and execute without a shell.

```bash
python tools/hicache_persistence/qualify_persistence.py \
  --config /path/to/runtime-persistence.json \
  --output /path/to/results \
  --mode full
```

The full run proves:

- empty-root write-through reaches the configured backup ratio;
- identical restart reselects A and restores at least the configured L3 ratio;
- concurrent restore returns every exact needle;
- representation B selects a different root and is cold;
- B does not modify A;
- rollback selects A again and restores its old prefix;
- the original representation is restarted in the final cleanup path.

For a machine reboot:

```bash
python tools/hicache_persistence/qualify_persistence.py \
  --config /path/to/runtime-persistence.json \
  --output /path/to/reboot-seed-results \
  --mode reboot-seed \
  --state-file /path/to/reboot-state.json

# Reboot through the normal operator path, then:

python tools/hicache_persistence/qualify_persistence.py \
  --config /path/to/runtime-persistence.json \
  --output /path/to/reboot-verify-results \
  --mode reboot-verify \
  --state-file /path/to/reboot-state.json
```

## Acceptance boundary

Pass means exact outputs, observed storage-hit ratios, namespace isolation, rollback reuse, successful concurrency, and a healthy original runtime after the run. Timing alone is not proof of persistence.

Interrupted NIXL FILE publication is not proven failure-atomic. A missing or rejected entry may become an ordinary cache miss and recomputation. Do not document transactional durability, exactly-once writes, or inference failure on cache-write failure.

## Removal criteria

- Remove patch 1 when upstream chunks every bounce-backed v2 pool transfer to registered capacity and retains equivalent large-pool read/write tests.
- Remove patch 2 when upstream guarantees disjoint IDs for concurrent FILE path registrations, or NIXL removes the unique-active-`devId` requirement, with concurrency coverage.
- Keep or replace the namespace helper until upstream provides an equally conservative representation identity and fail-closed manifest contract.
- Never restore the obsolete Mooncake patch as a fallback.
