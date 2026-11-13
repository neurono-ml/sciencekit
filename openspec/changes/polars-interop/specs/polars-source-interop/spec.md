# Spec Delta

## Purpose

Defines how Polars eager and lazy tables enter sciencekit as in-memory views or streaming sources,
and how sciencekit results return as Polars frames, including dtype mapping, null policy, column
projection, and streaming collection.

## ADDED Requirements

### Requirement: Eager Polars frames enter as tabular batches

The adapter SHALL convert a Polars `DataFrame` and a `Series` into the neutral tabular batch,
preserving column names, order, and values, and SHALL support selection by feature and target role.

#### Scenario: DataFrame becomes a tabular batch
- **WHEN** a `DataFrame` with numeric and string columns is imported
- **THEN** the tabular batch carries the same names, rows, and values

#### Scenario: Series becomes a single-column batch
- **WHEN** a `Series` is passed as features
- **THEN** the resulting batch has one column named after the series

### Requirement: Single-chunk primitive columns are imported without copying

The adapter SHALL borrow a Polars primitive column when it has exactly one chunk, no nulls, and a
dtype matching the model scalar; otherwise it SHALL materialize a copy and SHALL report the
conversion as copied.

#### Scenario: Contiguous column is borrowed
- **WHEN** a one-chunk `Float64` column without nulls feeds an `f64` model
- **THEN** the resulting view aliases the Polars buffer

#### Scenario: Multi-chunk column is materialized
- **WHEN** the same column is split into several chunks
- **THEN** the adapter concatenates the chunks into an owned block

### Requirement: Lazy frames become reopenable streaming sources

The adapter SHALL execute a lazy frame with Polars' streaming engine and a configured chunk size,
delivering batches through the tabular source contract, and SHALL reopen the query for each pass
when the frame is cloneable.

#### Scenario: Lazy source respects chunk size
- **WHEN** a lazy frame is imported with a chunk size of 2 048 rows
- **THEN** every delivered batch has at most 2 048 rows and the final batch is marked final

#### Scenario: Lazy and eager agree
- **WHEN** the same frame is fitted eagerly and through the lazy streaming source
- **THEN** the model parameters agree within floating-point tolerance

### Requirement: Selection is pushed into the Polars plan

The adapter SHALL project only the selected feature and target columns before executing a lazy
frame, so unselected columns are not materialized.

#### Scenario: Unselected column is never read
- **WHEN** a lazy frame with ten columns is imported selecting three
- **THEN** the executed plan outputs only those three columns in selection order

### Requirement: Polars dtypes map to neutral column types

The adapter SHALL map `Float32`, `Float64`, `Int64`, `Boolean`, `String`, `Categorical`, and `Enum`
to neutral columns, SHALL map nulls through the declared missing-value policy, and SHALL reject any
other dtype with a structured error naming the column and a suggested cast.

#### Scenario: Categorical column keeps its categories
- **WHEN** a categorical target column is imported
- **THEN** the label table matches the category values and round-trips

#### Scenario: Unsupported dtype is rejected
- **WHEN** a date or list column is selected as a numeric feature
- **THEN** import fails before execution with an unsupported-column-type error

### Requirement: Sciencekit results export to Polars

The adapter SHALL export dense matrices, CSR matrices, and streaming batches to Polars
`DataFrame`/`Series`, preserving values, names, and row count.

#### Scenario: Predictions become a series
- **WHEN** a prediction vector is exported
- **THEN** the result is an `f32` or `f64` series with the same values and length

#### Scenario: CSR block exports as sparse columns
- **WHEN** a CSR feature block is exported
- **THEN** row pointers, column indices, and values are preserved in the exported representation

### Requirement: Both scalar precisions are supported

Every Polars conversion SHALL support `f32` and `f64` models, casting explicitly when the Polars
dtype differs from the model scalar.

#### Scenario: f32 frame with f64 model
- **WHEN** an `Float32` frame is passed to an `f64` estimator
- **THEN** values are promoted and the conversion reports a cast
