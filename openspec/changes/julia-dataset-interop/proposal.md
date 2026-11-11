## Why

Julia's scientific stack is built around column-accessible tables (`NamedTuple` columns,
`DataFrames.jl`, `Arrow.jl`, `CSV.File`, `TypedTables.Table`) and column-major dense arrays
(`Matrix{Float64}`, `Matrix{Float32}`). Following ADR #285 and ADR #306, non-Rust language
bindings live in dedicated repositories under `neurono-ml` and call the shared `sciencekit_ffi`
C ABI rather than introducing a second toolchain or a language-specific ABI crate into the main
`neurono-ml/sciencekit` Cargo workspace.

## What Changes

- **External Julia package `ScienceKit.jl` in [`neurono-ml/sciencekit-jl`](https://github.com/neurono-ml/sciencekit-jl)**
  calling the shared `crates/sciencekit_ffi` C ABI (`1.x`) via `Libdl` and `ccall`. No Julia
  code is stored in `neurono-ml/sciencekit`.
- **Vendored C ABI header + native build script**: `include/sciencekit.h` is synced from
  `crates/sciencekit_ffi/include/sciencekit.h` (`scripts/sync-header.sh --check` in CI) and
  `scripts/build-native-library.sh` builds `libsciencekit_ffi` from a `neurono-ml/sciencekit`
  checkout.
- **Dense matrices and vectors**: `AbstractMatrix{Float64}`/`AbstractMatrix{Float32}` and
  `AbstractVector{Float64}`/`AbstractVector{Float32}` are validated and converted for the
  row-major ABI with explicit `ConversionReport` layout reporting (`:borrowed_contiguous` for
  1-row, 1-column, and `transpose(::Matrix)` inputs; `:transposed_copy` for general
  column-major matrices) under `GC.@preserve`.
- **Columnar table ingestion**: `extract_table` and `fit_table` accept any column-accessible
  table (`NamedTuple` columns, `Vector{<:NamedTuple}`, `Dict`, `DataFrame`, `CSV.File`,
  `TypedTables.Table`) with `:reject`, `:nan`, and `:fill => value` missingness policies.
- **Handle lifecycle and structured errors**: `Builder` and `Model` wrap opaque handles with a
  non-yielding `Threads.SpinLock` lease counter, idempotent `close!`, GC finalizers, `save`/`load`
  persistence, and `ScienceKitError(status, code, message)`.
- **Documentation site**: English mdBook on branch `docs/documentations` published to
  <https://neurono-ml.github.io/sciencekit-jl/> with generated `sitemap.xml`.

## Capabilities

### New Capabilities

- `julia-dataset-interop`: Julia package, dense array and columnar table conversions, layout
  reporting, handle lifecycle, persistence, and structured error mapping in
  `neurono-ml/sciencekit-jl` over `sciencekit_ffi`.

### Modified Capabilities

<!-- none -->

## Impact

- **Code location:** External repository `neurono-ml/sciencekit-jl` (`Project.toml`, `src/`,
  `test/`, `include/sciencekit.h`, `scripts/`, `docs/`, `.github/workflows/`); zero Rust
  workspace changes in `neurono-ml/sciencekit`.
- **Dependencies (Julia):** `Libdl` (stdlib); `Test` (stdlib, test-only).
- **Dependencies (Rust):** Existing `crates/sciencekit_ffi` (`cdylib`) in `neurono-ml/sciencekit`.
- **Acceptance (PRD §8.7/§10.3):** Verified in `sciencekit-devcontainer:latest` (Julia 1.10.5)
  and GitHub Actions CI across small and large fixtures, `Float32`/`Float64`, concurrent
  `Threads.@spawn` tasks, persistence round-trip, and repeated-failure leak checks.
