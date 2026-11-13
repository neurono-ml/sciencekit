> **Implementation repository:** [`neurono-ml/sciencekit-jl`](https://github.com/neurono-ml/sciencekit-jl)
> (documentation: <https://neurono-ml.github.io/sciencekit-jl/>, ADR: #306). Per ADR #285, no Julia
> source files live in `neurono-ml/sciencekit`; the package consumes `crates/sciencekit_ffi` over the
> stable C ABI.

## 1. ABI consumption and package skeleton (`neurono-ml/sciencekit-jl`)

- [x] 1.1 Vendor `include/sciencekit.h` from `crates/sciencekit_ffi/include/sciencekit.h` with `scripts/sync-header.sh` and `scripts/build-native-library.sh`.
- [x] 1.2 Create `ScienceKit.jl` (`Project.toml`, `src/ScienceKit.jl`, `__init__()`, `Libdl` symbol resolution, ABI version check).
- [x] 1.3 TDD ABI error path: status codes + thread-local `sk_ffi_last_error` map to `ScienceKitError(status, code, message)`.
- [x] 1.4 TDD handle and output-array ownership: non-yielding `Threads.SpinLock` lease counter, idempotent `close!`, GC finalizers, `sk_ffi_array_free` in `finally`, and zero leaks across repeated failures.

## 2. Columnar table ingestion

- [x] 2.1 TDD `extract_table` and `fit_table` for column-accessible tables (`NamedTuple`, row tables, `Dict`, and property-exposing tables): column names, numeric conversion, and feature/target selection.
- [x] 2.2 TDD `KeyError` on unknown columns and `DimensionMismatch` on uneven column lengths.
- [x] 2.3 TDD `Union{T, Missing}` columns under `:reject`, `:nan`, and `:fill => value` policies.

## 3. Native dense arrays and estimator operations

- [x] 3.1 TDD `Matrix{Float32}`/`Matrix{Float64}` and `SubArray`/`transpose` views: row/column layout semantics, `ConversionReport` (`:borrowed_contiguous` vs `:transposed_copy`), and `GC.@preserve`.
- [x] 3.2 TDD wrong element type (`Matrix{Int64}`) raises `ScienceKitError(STATUS_INVALID_ARGUMENT, ...)` with float conversion hint; empty dimensions raise `STATUS_SHAPE`.
- [x] 3.3 TDD `fit`, `predict`, `score` (`LinearRegression` in `Float64` and `Float32`), `transform` (`StandardScaler`), and `save`/`load` persistence round-trip.

## 4. Concurrency, CI, documentation, and bookkeeping

- [x] 4.1 TDD concurrent `Threads.@spawn` predictions on a shared `Model` and concurrent fits on independent `Builder`s, plus `4_000 x 8` large-array verification.
- [x] 4.2 Add GitHub Actions CI (`.github/workflows/ci.yml`) installing Julia 1.10.5 from the official tarball, building `sciencekit_ffi` from `neurono-ml/sciencekit`, checking header sync, and running `Pkg.test()`; verify in `sciencekit-devcontainer:latest`.
- [x] 4.3 Publish the English mdBook documentation site on `docs/documentations` to GitHub Pages (`https://neurono-ml.github.io/sciencekit-jl/`) with `sitemap.xml` (`docs/scripts/postbuild.mjs`).
- [x] 4.4 Create ADR issue #306 and record external repository status on `feat/dataset-interop-openspec` (PR #221).
- [ ] 4.5 Follow-up — **DEFERRED (2026-11-13)**: native `SparseMatrixCSC` CSR bridge and zero-copy `Arrow.Table` C stream ingestion. Blocked on the upstream C ABI: `sciencekit_ffi` at `origin/main` (6140ba1) exposes neither sparse CSR nor `ArrowArrayStream` entry points (`git grep -n "ArrowArrayStream\|csr" origin/main -- crates/sciencekit_ffi` matches only the `representation: "csr"` error-test string in `src/status_tests.rs`; `include/sciencekit.h` has no `Arrow`/`Sparse` declarations). Re-scope this task when the FFI publishes those descriptors.
