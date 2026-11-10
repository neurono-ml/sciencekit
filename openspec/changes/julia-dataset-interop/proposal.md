## Why

Julia's scientific stack is built around the Tables.jl ecosystem: `DataFrames.jl`, `Arrow.jl`,
`CSV.jl`, `TypedTables.jl`, `JuliaDB.jl`, and many others all implement the same column-access
interface. A sciencekit Julia binding should accept **any Tables.jl source** plus native
`AbstractMatrix` and `SparseMatrixCSC` values, instead of supporting each package individually.

Julia also has `CategoricalArray` and `Union{T, Missing}` columns, which need the same missing-value
and label semantics as the Rust and Python paths.

## What Changes

- **New Julia package `ScienceKit.jl`** and the Rust cdylib it calls through a C ABI.
- **Tables.jl ingestion**: any source implementing `Tables.istable` is accepted by calling
  `Tables.columns`, reading column names and types, and converting to the neutral tabular batch;
  this covers `DataFrames.jl`, `Arrow.Table`, `CSV.File`, `TypedTables.Table`, and `JuliaDB`.
- **Native arrays**: `AbstractMatrix{<:Real}` is accepted, with `Matrix{Float32}`/`Matrix{Float64}`
  borrowed zero-copy when column-major layout permits a row view (Julia is column-major, so a
  row-major sciencekit view requires a transpose or a copy; the binding documents and reports this).
- **Sparse arrays**: `SparseMatrixCSC` is converted to CSR explicitly or exposed through a CSR view
  by transposing index arrays; stored zeros are preserved.
- **Categorical arrays**: `CategoricalArray`/`CategoricalVector` map to the label table and
  category codes; `missing` maps to the missing-value policy.
- **Arrow C Data Interface**: `Arrow.Table` and `Arrow.Stream` are transferred through the C stream
  when the Julia Arrow version supports it; otherwise through Arrow IPC bytes, with a documented
  fallback and no silent full-table copies.
- **Return values**: results return as Julia `Matrix`, `SparseMatrixCSC`, `Vector`, `DataFrame`, or
  any Tables.jl table, selectable by the caller.
- **Errors**: structured Julia exceptions (`ArgumentError`, `DimensionMismatch`, `KeyError`) with
  the column name and a suggested conversion.
- **Tests**: Tables.jl sources, matrices, sparse, categorical/missing, large and small data,
  threading through `Threads.@threads`, `f32`/`f64`.

## Capabilities

### New Capabilities

- `julia-dataset-interop`: Julia table and array representations (Tables.jl, DataFrames, Arrow.jl,
  matrices, sparse, categorical) as sciencekit inputs and outputs, with layout-aware zero-copy and
  missing-value semantics.

### Modified Capabilities

<!-- none -->

## Impact

- **New code:** `julia/ScienceKit.jl` package (Project.toml, src, test) and
  `crates/sciencekit_julia_abi` (C ABI cdylib) or an extension of the existing bridge crate.
- **Dependencies (Julia):** `Tables`, `DataFrames`, `Arrow`, `SparseArrays`, `CategoricalArrays`;
  test-only `CSV`, `TypedTables`.
- **Dependencies (Rust):** `sciencekit_interop` with `arrow`; the C ABI uses Arrow C Data Interface.
- **Sequencing:** depends on `interop-foundation`, `arrow-interop`, and the binding ABI. The PRD
  currently lists only Python bindings; this change also updates the PRD's binding roadmap.
- **Acceptance (PRD §8.7/§10.3):** every representation fits the same estimator with equal results;
  large and small data; threaded calls; `f32`/`f64`; model export and metrics unaffected.
