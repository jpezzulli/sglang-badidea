# Frozen Pennyroyal HiCache/NIXL persistence audit

Audit date: 2026-08-26

Frozen runtime branch: `pennyroyal-main-sm120-final` at `8e197ed3afc559f29562a2e7de9026f011f5d28f`

Frozen upstream base: `5a7b26c636deb2def43640bab6c63146dbe536dc`

Current SGLang upstream inspected: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`

Installed NIXL: `1.4.0`, clean upstream commit `aecbc3846d92c34c7507a58d776e1fda50ff4fba`

Current NIXL upstream inspected: `f71a468f6790c1c519db37f7508a5dca2dc98976`

## Scope

This audit includes only persistent target KV, hybrid Mamba/GDN state, packed or sidecar draft state, NIXL FILE routing, representation namespaces, checkpoint/path identity, and failure behavior that can change whether a prefix is reusable after eviction or restart.

Qwen3.8 decode kernels, XQA, DFlash candidate selection, ragged verification, CUDA compiler selection, and request-timing work are excluded unless persistence directly calls the changed code. None of those excluded changes is required by the persistence patch series.

## Classification

| Change or behavior | Classification | Carry forward? | Evidence and disposition |
|---|---|---:|---|
| Hybrid target-KV plus Mamba/GDN state offload and `HybridCacheController` | `upstreamed` | No local patch | Landed through SGLang `0986bed8e2` (#20457) and later unified-cache work. The frozen tree did not modify these paths locally. |
| NIXL v2 hybrid-pool serialization and component-specific keys | `upstreamed` | No local patch | SGLang `978bce2063` (#29191) supplies hybrid NIXL support. Current `HiCacheNixl` derives separate Mamba temporal/conv and K/V component keys. |
| Packed/sidecar draft persistence (`PoolName.DRAFT`) | `upstreamed` | No local patch | SGLang `8e11feb68e` (#30393) supplies draft offload. This is the only DFlash-related functionality persistence needs; DFlash decode performance patches are irrelevant. |
| DeepSeek-V4/HostPoolGroup multi-pool handling | `upstreamed` | No local patch | SGLang `a34f81251f` (#30762), followed by current host-pool refactor `aa718f7343` (#36232). |
| Mamba retained-prefix tracking/branching corrections | `upstreamed` | No local patch | SGLang `8f3d3a31f4` (#29792) and `3c533acec6` (#33639). Runtime state-count limits remain launcher policy. |
| Cleaner grouping for hybrid NIXL component keys | `upstreamed` | No local patch | SGLang `8bb106cee9` (#35130). |
| Bound v2 bounce-backed NIXL transfers to the registered staging capacity | `partially upstreamed` | **Patch 1** | Upstream allocates a fixed `STORAGE_BATCH_SIZE` bounce area and already batches v1 I/O, but current v2 hybrid I/O still describes an arbitrarily long transfer over that fixed buffer. Frozen commit `8e197ed3af` supplies the missing logical-page split. |
| Disjoint device IDs for overlapping path-mode FILE registrations | `partially upstreamed` | **Patch 2** | Upstream already allocates monotonically unique IDs for OBJ registration, but FILE path mode restarts at `1` for every registration. Current NIXL still states that every active path-mode file needs a unique `devId` and rejects reuse. Frozen commit `8e197ed3af` extends the locked allocator to FILE. |
| Select NIXL FILE base directories through `SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR` | `upstreamed` | Configuration only | `NixlFileManager` routes existence queries, reads, writes, clear, and cleaner roots through the selected base directory. |
| Deterministic representation namespace and fail-closed manifest | `local-only` | **Patch 3 toolkit** | `derive_namespace.py` is launch policy, not SGLang core. It derives a readable root plus 12 hex SHA-256 characters and verifies the full identity in `namespace-identity.json`. |
| Checkpoint content identity, including HF revision/LFS digest and full-payload fallback | `local-only` | **Patch 3 toolkit** | Added after review found that path, size, header, and mtime alone could reuse cache after payload replacement. This remains deliberately conservative local policy. |
| Namespace identity field selection in the Pennyroyal launcher | `local-only` | Port by configuration | Model/checkpoint, runtime revision/diff, context, topology, page/layout geometry, cache dtypes, draft shape, Mamba dtypes, backends, Torch, and SM architecture are supplied to the generic helper. Host paths and capacities are not hard-coded into the helper. |
| NIXL POSIX build for Python 3.12/CUDA 13/SM120/GCC 15 | `upstreamed` | Rebuild, do not patch | The installed NIXL source tree is clean upstream. No NIXL source patch is part of Pennyroyal. Rebuild current NIXL for the future Python/CUDA/toolchain compatibility unit. |
| POSIX/O_DIRECT/io_uring configuration and 68/65 cleaner watermarks | `local-only` | Re-evaluate per filesystem | These are host/storage policy. Watermarks are filesystem occupancy percentages, not an exact namespace byte quota. |
| `kernel + page_first`, write-through, timeout prefetch, host-memory size | `local-only` | Re-qualify | Runtime configuration, not a source delta. `direct + page_first_direct` was rejected for the frozen representation. |
| Mamba cache size and states-per-path limits | `local-only` | Re-profile/re-qualify | They control live allocator/COW headroom but do not change the serialized bytes of one Mamba state. They therefore are not required namespace fields unless upstream changes that contract. Current Pennyroyal uses 5 states/path and 24 entries. |
| Frozen Mooncake capacity/retry patch `ba600c682a` | `obsolete` | **Never carry** | Mooncake was rejected and removed. It is not a fallback; the fallback is the same runtime without HiCache RAM/SSD tiers. |
| Temporary calibrated FP8 KV scale shard | `obsolete` | Never carry | Removed after the A/B. Default target and draft unit scales are the selected representation. A future non-unit scale is representation-sensitive and must select another namespace. |

## Directly established invariants

1. A reusable hybrid prefix is bounded by the shortest complete required pool. Target KV alone is insufficient when Mamba/GDN or draft state is required.
2. Each logical page's physical component keys are pool-qualified. Mamba temporal and convolution tensors, and draft K/V components, cannot alias target KV or one another.
3. A bounce-backed transfer may describe no more logical pages than its registered staging buffer. Keys, page-expanded host indices, buffer slots, copy-back offsets, and page results must be sliced on the same boundary.
4. Every simultaneously active NIXL path-mode FILE registration must use a disjoint device-ID range.
5. Every NIXL FILE lookup and cleaner scan must stay beneath the selected representation root.
6. A namespace identity mismatch fails launch instead of silently reusing files.
7. Returning to the exact A representation must derive the exact A root; a B representation must derive a different root and leave A unchanged.
8. Restart recovery is discovered from deterministic page hashes and FILE existence. GPU/host buffers and radix metadata are process-local.
9. Missing or unqueryable state may become a cache miss and ordinary recomputation. Cache-write failure is not an inference failure.

## Remaining upstream limitation

NIXL FILE path mode still does not establish temporary-file publication or a completion marker that makes an interrupted write distinguishable from a complete file solely by pathname. SGLang also does not perform a strong content checksum before using every restored page. Cache state is disposable, so the accepted response is exact-output validation plus recomputation on a miss—not a claim of transactional or failure-atomic durability.

## Excluded frozen commits

The following frozen commits are intentionally absent from the persistence series: independent draft configuration hooks, draftless custom algorithm hooks, GCC/NVCC selection, request phase timing, XQA speculative masks, DFlash2 selector/convolution work, and ragged verification/mRoPE work. They may matter to a particular future model, but persistence does not depend on them.
