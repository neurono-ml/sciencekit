# Spec Delta

## Purpose

Defines the external `ScienceKit.jl` package (`neurono-ml/sciencekit-jl`) over the shared
`sciencekit_ffi` C ABI: module initialization, dense matrix and columnar table conversions,
layout reporting, estimator operations, model persistence, handle lifecycle, threading, and
structured error mapping.

## ADDED Requirements

### Requirement: Package loads `sciencekit_ffi` and verifies ABI major version

The Julia package in `neurono-ml/sciencekit-jl` SHALL load `libsciencekit_ffi` during
`__init__()`, verify ABI major-version compatibility, and expose version and feature queries.

#### Scenario: Module initialization and version check
- **WHEN** Julia executes `using ScienceKit` with `libsciencekit_ffi` available
- **THEN** `abi_version().major == 1` and `features()` advertises the compiled estimator and precision flags

### Requirement: Native matrices are accepted with documented layout behavior

The binding SHALL accept `AbstractMatrix` and `AbstractVector` of `Float32` and `Float64`,
convert or borrow buffers safely under `GC.@preserve` for `sciencekit_ffi`'s row-major ABI,
report the conversion path via `ConversionReport`, and reject unsupported dtypes or empty shapes.

#### Scenario: Column-major matrix is accepted and reported
- **WHEN** a multi-column `Matrix{Float64}` is passed as features
- **THEN** rows map to samples, columns map to features, and `inspect_matrix_conversion` reports `:transposed_copy`

#### Scenario: Transposed matrix wrapper is borrowed zero-copy
- **WHEN** `transpose(parent)` of a dense `(cols, rows)` `Matrix{Float64}` is passed
- **THEN** `inspect_matrix_conversion` reports `:borrowed_contiguous` with `copied == false`

#### Scenario: Wrong element type is rejected
- **WHEN** a `Matrix{Int64}` is passed without an explicit float conversion
- **THEN** the binding raises `ScienceKitError` with `STATUS_INVALID_ARGUMENT` suggesting `Float64.(x)` or `Float32.(x)`

### Requirement: Column-accessible tables and missing-value policies are supported

The binding SHALL extract numeric feature matrices and target vectors from column-accessible
tables (`NamedTuple`, row tables, `Dict`, and property-exposing tables such as `DataFrame`,
`CSV.File`, and `TypedTables.Table`) and apply `:reject`, `:nan`, or `:fill => value` policies
to `missing` entries.

#### Scenario: Named columns and ignored non-numeric columns
- **WHEN** `extract_table` or `fit_table` is called with explicit `features` and `target` names on a table that also has string columns
- **THEN** the selected numeric columns produce the expected `Matrix` and `Vector` values

#### Scenario: Missing value under reject policy
- **WHEN** a selected column contains `missing` under `missing_policy=:reject`
- **THEN** the call raises `ScienceKitError` with `STATUS_INVALID_ARGUMENT` naming the column

#### Scenario: Unknown column or uneven column lengths
- **WHEN** an absent column name is requested or column lengths disagree
- **THEN** the binding raises `KeyError` for unknown columns and `DimensionMismatch` for uneven lengths

### Requirement: Operations and persistence round-trip across the C ABI

Builders and fitted models SHALL expose `fit`, `fit_table`, `predict`, `transform`, `score`,
`save`, and `load` for `Float64` and `Float32` configurations supported by `sciencekit_ffi`.

#### Scenario: Linear regression fit, predict, score, and persistence round-trip
- **WHEN** a `Float64` linear regression builder is fitted on a linear fixture, saved to disk, and reloaded via `load`
- **THEN** `predict` returns `Vector{Float64}` matching targets, `score` returns `1.0`, and the reloaded model reproduces predictions after the original handles are closed

#### Scenario: Standard scaler transform preserves matrix geometry
- **WHEN** a `StandardScaler` builder is fitted and `transform` is called on a `(rows, cols)` matrix
- **THEN** `transform` returns a column-major `Matrix` of shape `(rows, cols)` with centered, unit-variance columns

### Requirement: Lifecycle safety, structured errors, and concurrency

Handles SHALL release native resources deterministically via idempotent `close!` and GC
finalizers without use-after-free or handle leaks, map ABI failures to `ScienceKitError(status, code, message)`,
and support concurrent multi-threaded calls.

#### Scenario: Idempotent close and closed-handle protection
- **WHEN** `close!` is called multiple times on a `Builder` or `Model` and an operation is then invoked on it
- **THEN** repeated `close!` calls succeed as no-ops and the subsequent operation raises `ScienceKitError` with `STATUS_INVALID_HANDLE`

#### Scenario: Repeated failures leak zero handles
- **WHEN** invalid operations (`fit`, `predict`, `transform`, `score`, `load`, closed handles) fail repeatedly across hundreds of iterations
- **THEN** `live_handle_count()` returns to its baseline with zero leaked handles

#### Scenario: Concurrent multi-threaded predictions and fits
- **WHEN** multiple Julia tasks (`Threads.@spawn`) run `predict` on a shared `Model` and `fit` on independent `Builder` instances
- **THEN** every task produces accurate results without data races or handle corruption
