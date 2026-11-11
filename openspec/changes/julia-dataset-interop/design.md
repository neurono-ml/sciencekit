# Design

## Context

See `proposal.md` — Why, ADR #285, and ADR #306. Keeping `ScienceKit.jl` in its own repository
(`neurono-ml/sciencekit-jl`) preserves `neurono-ml/sciencekit` as a pure Rust Cargo workspace
while letting Julia users consume a standard `Project.toml` package over the shared
`crates/sciencekit_ffi` C ABI.

Julia matrices are column-major, whereas `SKFfiArray` descriptors in `sciencekit_ffi` require
row-major layout. Julia also forbids task switches (`ReentrantLock`) inside garbage-collector
finalizers, while multi-threaded callers may invoke `predict` concurrently or race with `close!`.

## Goals / Non-Goals

**Goals**

- External repository `neurono-ml/sciencekit-jl` with zero non-stdlib runtime dependencies (`Libdl` only).
- Layout-honest conversions (`ConversionReport`) and `GC.@preserve` safety across every `ccall`.
- Column-accessible table ingestion (`extract_table`, `fit_table`) with `:reject`, `:nan`, and `:fill => value` policies.
- Deterministic `close!` + GC-safe finalizers with zero use-after-free and zero handle leaks.
- Dedicated English mdBook documentation site on GitHub Pages (`https://neurono-ml.github.io/sciencekit-jl/`).

**Non-Goals**

- No Julia code or separate `sciencekit_julia_abi` crate in `neurono-ml/sciencekit`.
- No `jlrs` embedding.
- Sparse CSC↔CSR and Arrow C Data Interface streaming entry points are deferred until `sciencekit_ffi` exposes sparse and Arrow stream descriptors on the shared C ABI.

## Decisions

### Decision 1 — External repository + vendored C header + `Libdl` discovery

`neurono-ml/sciencekit-jl` vendors `include/sciencekit.h` (checked against `crates/sciencekit_ffi/include/sciencekit.h`
in CI with `scripts/sync-header.sh --check`) and provides `scripts/build-native-library.sh` to
compile `cargo build -p sciencekit_ffi --release` into `lib/`. At runtime, `ScienceKit.__init__()`
discovers `libsciencekit_ffi` (`SCIENCEKIT_FFI_LIB`, `SCIENCEKIT_LIB_DIR`, `lib/`, system loader),
caches function pointers with `Libdl.dlsym`, and checks `abi_version().major == 1`.

### Decision 2 — Non-yielding `Threads.SpinLock` lease lifecycle

`Builder` and `Model` wrap opaque `Ptr{Cvoid}` handles guarded by a `Threads.SpinLock` and an
`in_flight` lease counter:
- Active `ccall` operations increment `in_flight` before reading the pointer and decrement it in `finally`.
- `close!` is idempotent: if `in_flight == 0`, it frees the native handle immediately; if `in_flight > 0`, it marks `closed = true` and the last exiting operation frees the handle.
- Because `SpinLock` never yields tasks, GC finalizers safely invoke the same close path without triggering Julia's finalizer task-switch error.

### Decision 3 — Layout-aware dense array and table conversion

- 1-row `Matrix{T}`, 1-column `Matrix{T}`, and `transpose(::Matrix{T})` wrappers whose parent has shape `(cols, rows)` are borrowed zero-copy (`:borrowed_contiguous`).
- General multi-column column-major `AbstractMatrix{T}` inputs transpose into a row-major buffer via `permutedims` (`:transposed_copy`).
- Native output descriptors (`SKFfiArray`) are copied into Julia `Vector{T}` or column-major `Matrix{T}` inside `try ... finally sk_ffi_array_free(...) end`.
- `extract_table` reads column-accessible tables (`NamedTuple`, row vectors, `Dict`, `DataFrame`, `CSV.File`, `TypedTables.Table`) and enforces column selection and missingness policies.

### Decision 4 — Structured `ScienceKitError`

`ScienceKitError <: Exception` carries `status::Status`, numeric `code::Int32`, and the copied
thread-local `sk_ffi_last_error` `message::String`. Table lookup and shape mismatches raise
`KeyError`, `DimensionMismatch`, or `ScienceKitError(STATUS_SHAPE | STATUS_INVALID_ARGUMENT, ...)`.

## Risks / Trade-offs

- **Native build prerequisite** → Mitigated by `scripts/build-native-library.sh`, `SCIENCEKIT_FFI_LIB`, and automated GitHub Actions CI building `sciencekit_ffi` from `neurono-ml/sciencekit`.
- **Column-major → row-major copy** → Mitigated by zero-copy detection for `transpose(::Matrix)` and single-row/single-column matrices plus explicit reporting via `inspect_matrix_conversion`.

## Migration Plan

Additive external repository (`neurono-ml/sciencekit-jl`) and documentation site; no breaking changes to `neurono-ml/sciencekit`.
