## 1. ABI and package skeleton

- [ ] 1.1 Create `crates/sciencekit_julia_abi` (cdylib) exposing status-code functions and Arrow C stream import/export; panic containment with `catch_unwind`.
- [ ] 1.2 Create `julia/ScienceKit.jl` with `Project.toml`, module layout, and a minimal `fit`/`predict`/`transform` calling the ABI.
- [ ] 1.3 TDD ABI error path: Rust errors cross as status + message and become Julia exceptions.
- [ ] 1.4 TDD ABI ownership: input/output Arrow streams released exactly once; no leak under repeated calls.

## 2. Tables.jl ingestion

- [ ] 2.1 TDD a generic Tables.jl source: names, column types, values, selection by name.
- [ ] 2.2 TDD DataFrames.jl source and CSV.File/TypedTables sources (test-only dependencies).
- [ ] 2.3 TDD `Union{T,Missing}` columns under reject/NaN/fill policies.
- [ ] 2.4 TDD string columns as nominal features/targets and as categorical dictionaries.

## 3. Native and sparse arrays

- [ ] 3.1 TDD `Matrix{Float32}`/`Matrix{Float64}`: row/column interpretation, conversion path reporting, transpose correctness.
- [ ] 3.2 TDD wrong element type (`Matrix{Int}`) raises `ArgumentError`; explicit cast path works.
- [ ] 3.3 TDD `SparseMatrixCSC` → CSR: shape, values, stored zeros; conversion reporting.
- [ ] 3.4 TDD sparse fit/transform and CSC result reconstruction.

## 4. Categorical, Arrow, results, threading

- [ ] 4.1 TDD `CategoricalVector`/`CategoricalArray` target: label table and prediction round-trip.
- [ ] 4.2 TDD `Arrow.Table` through the C Data Interface when supported; assert bounded memory.
- [ ] 4.3 TDD Arrow IPC fallback path and its reporting.
- [ ] 4.4 TDD result formats: `Vector`, `Matrix`, `SparseMatrixCSC`, `DataFrame`, generic Tables.jl.
- [ ] 4.5 TDD `Threads.@threads` concurrent fits on independent data.
- [ ] 4.6 TDD `Float32`/`Float64` and large/small data across all representations.
- [ ] 4.7 Run Rust CI gates plus `julia --project -e 'using Pkg; Pkg.test()'`.
- [ ] 4.8 Add the Julia interop book chapter and the `Development Explained` chapter on the C ABI; register in `SUMMARY.md`; add Julia docs.
- [ ] 4.9 Update the PRD binding roadmap and `CHANGELOG.md`; open the ADR issue for this change.
