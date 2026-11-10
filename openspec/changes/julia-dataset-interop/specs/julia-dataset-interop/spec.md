# Spec Delta

## Purpose

Defines which Julia table and array representations the sciencekit binding accepts and returns, how
Tables.jl sources, matrices, sparse matrices, and categorical/missing columns are converted, and
where zero-copy is possible given Julia's column-major layout.

## ADDED Requirements

### Requirement: Any Tables.jl source is accepted

The binding SHALL accept any object implementing the Tables.jl interface by reading its columns,
names, and element types, converting it to the neutral tabular batch, and applying feature/target
selection by name.

#### Scenario: DataFrames.jl frame is accepted
- **WHEN** a `DataFrame` with numeric and string columns is passed
- **THEN** the binding sees the same names, rows, and values

#### Scenario: CSV.File and TypedTables are accepted
- **WHEN** a `CSV.File` or `TypedTables.Table` is passed
- **THEN** it is accepted without a package-specific code path

### Requirement: Native matrices are accepted with documented layout behavior

The binding SHALL accept `AbstractMatrix{<:Real}` of `Float32` and `Float64`. Because Julia stores
matrices column-major, the binding SHALL either transpose into a row-major owned block or expose a
field-transposed view, and SHALL report which path was taken.

#### Scenario: Column-major matrix is accepted
- **WHEN** a `Matrix{Float64}` is passed as features
- **THEN** the values are interpreted as rows of samples and columns of features, and the conversion path is reported

#### Scenario: Wrong element type
- **WHEN** a `Matrix{Int64}` is passed without an explicit cast
- **THEN** the binding raises an `ArgumentError` suggesting a float conversion

### Requirement: Sparse matrices map to CSR

The binding SHALL accept `SparseMatrixCSC` and convert it to CSR by transposing index arrays,
preserving stored zeros; it SHALL report the conversion and reject unsupported element types.

#### Scenario: CSC to CSR preserves values
- **WHEN** a `SparseMatrixCSC{Float64}` is passed
- **THEN** the resulting CSR matrix has the same logical values and shape

#### Scenario: Stored zero is preserved
- **WHEN** the sparse matrix stores an explicit zero
- **THEN** the converted matrix keeps that entry

### Requirement: Missing and categorical values follow the shared policy

The binding SHALL map `missing` in Tables.jl columns and matrices through the declared
missing-value policy, and SHALL map `CategoricalArray`/`CategoricalVector` to category codes and a
reversible label table.

#### Scenario: missing under reject
- **WHEN** a column contains `missing` under the reject policy
- **THEN** the call raises an error naming the column

#### Scenario: Categorical target round-trips
- **WHEN** a `CategoricalVector` target is used to fit a classifier
- **THEN** predictions map back to the original category values

### Requirement: Arrow.jl tables use the Arrow interchange

The binding SHALL accept `Arrow.Table` and `Arrow.Stream` through the Arrow C Data Interface when
the installed Arrow.jl supports it, and otherwise through Arrow IPC bytes, reporting which path was
used and never silently materializing the whole table.

#### Scenario: Arrow.Table is transferred without a full copy
- **WHEN** an `Arrow.Table` over a large file is passed
- **THEN** the binding consumes it batch by batch and resident memory stays bounded

#### Scenario: Fallback is reported
- **WHEN** the installed Arrow.jl lacks C Data Interface support
- **THEN** the binding uses the IPC fallback and reports it

### Requirement: Results return in the requested Julia representation

The binding SHALL return predictions and transforms as Julia vectors or matrices by default, and
SHALL offer `DataFrame` and generic Tables.jl outputs; values SHALL round-trip without loss.

#### Scenario: Default prediction type
- **WHEN** `predict` is called with no format option
- **THEN** the result is a `Vector{Float32}` or `Vector{Float64}`

#### Scenario: DataFrame result
- **WHEN** a DataFrame result is requested
- **THEN** columns have the expected names and values

### Requirement: Threaded calls are safe

The binding SHALL be safe to call from multiple Julia threads on independent data, with no shared
mutable conversion state.

#### Scenario: Threaded fits
- **WHEN** `Threads.@threads` runs two fits on separate matrices
- **THEN** both produce correct independent models without races

### Requirement: Errors are structured Julia exceptions

Every conversion failure SHALL raise a Julia exception with the column or array name and a
suggested conversion: `KeyError` for unknown columns, `DimensionMismatch` for shape errors, and
`ArgumentError` for unsupported types.

#### Scenario: Unknown column
- **WHEN** a feature name absent from the table is requested
- **THEN** the binding raises `KeyError` naming it

#### Scenario: Shape mismatch
- **WHEN** a target length differs from the feature row count
- **THEN** the binding raises `DimensionMismatch`

### Requirement: Both scalar precisions are supported

Every Julia conversion SHALL support `Float32` and `Float64` models, casting explicitly when the
element type differs and reporting the cast.

#### Scenario: Float32 table with Float64 model
- **WHEN** a `Float32` table is passed to an `f64` estimator
- **THEN** values are promoted and the conversion reports a cast
