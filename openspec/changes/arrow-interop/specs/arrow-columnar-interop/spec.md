# Spec Delta

## Purpose

Defines how Apache Arrow data — in-memory arrays and batches, streaming readers, the C Data
Interface, and columnar file formats — enters and leaves sciencekit's tabular seam, preserving
null semantics and using zero-copy only where Arrow's layout allows it.

## ADDED Requirements

### Requirement: Arrow in-memory tables enter the tabular seam

The adapter SHALL convert Arrow record batches and tables into the neutral tabular batch, preserving
column names, order, logical types, and row count. It SHALL accept a single record batch, a table of
chunks, and a batch reader.

#### Scenario: Record batch becomes a tabular batch
- **WHEN** a record batch with numeric and string columns is imported
- **THEN** the tabular batch has the same column names, row count, and values

#### Scenario: Chunked table is concatenated in order
- **WHEN** a table contains several chunks per column
- **THEN** the imported column preserves chunk order and total length

### Requirement: Primitive Arrow arrays are imported without copying where possible

The adapter SHALL return a zero-copy view when an Arrow primitive column has no nulls, one buffer,
and a dtype matching the model scalar; otherwise it SHALL materialize a copy and SHALL NOT report a
zero-copy conversion.

#### Scenario: Matching primitive column is borrowed
- **WHEN** an `f64` Arrow column with no nulls and one buffer is selected as the only feature
- **THEN** the resulting view aliases the Arrow buffer

#### Scenario: Nulls force materialization
- **WHEN** the same column carries a validity bitmap with nulls under the numeric NaN policy
- **THEN** the adapter allocates a new block and writes NaN at null positions

### Requirement: Arrow validity maps to the declared missing-value policy

The adapter SHALL translate Arrow nulls according to the selected policy — reject, numeric NaN,
categorical missing code, or constant fill — and SHALL reject integer nulls when promotion would
lose precision.

#### Scenario: Null string becomes the categorical sentinel
- **WHEN** a dictionary/string feature column has nulls under the categorical policy
- **THEN** the missing code occupies that position and no valid code is reused

#### Scenario: Reject policy reports the column
- **WHEN** a nullable Arrow feature column has a null under the reject policy
- **THEN** import fails with a null-in-column error naming the Arrow field

### Requirement: Dictionary and string arrays map to categorical data

The adapter SHALL map Arrow dictionary arrays and string arrays to nominal or category-coded
columns according to the column role, preserving the dictionary values as the label table.

#### Scenario: Dictionary target yields a reversible label table
- **WHEN** a dictionary-encoded target column is imported for a classifier
- **THEN** canonicalization produces a label table that restores the original labels

#### Scenario: String feature is not silently numeric
- **WHEN** a string column is selected as a feature of a numeric algorithm
- **THEN** import fails with an unsupported-column-type error suggesting categorical encoding

### Requirement: Arrow batch readers become reopenable tabular sources

The adapter SHALL expose an Arrow batch reader as a fallible tabular source delivering owned batches
of the configured maximum row count, and SHALL provide a reopenable supplier when the reader can be
recreated (file path or factory). Only selected columns SHALL be requested from the reader.

#### Scenario: Batch size is respected
- **WHEN** a reader over a 10 000-row table is configured with batch size 1 000
- **THEN** each delivered batch has at most 1 000 rows and the last batch is marked final

#### Scenario: Projection limits work
- **WHEN** the selection requests two of ten Parquet columns
- **THEN** the reader reads only those two columns and the returned batches carry them in selection order

### Requirement: C Data Interface import and export are ownership-safe

The adapter SHALL import `ArrowSchema`/`ArrowArray`/`ArrowArrayStream` structures and export
sciencekit arrays and streams through the same structures, transferring ownership exactly once and
invoking the release callback when the consumer releases, with no use-after-free or double free.

#### Scenario: Exported stream is consumed by another runtime
- **WHEN** a sciencekit data batch is exported as an `ArrowArrayStream` and consumed by an external Arrow implementation
- **THEN** the consumer sees the same schema and values and its release call frees the buffers exactly once

#### Scenario: Imported array survives producer drop
- **WHEN** an imported Arrow array is retained after the producer releases its handle
- **THEN** the data remains valid because the adapter took ownership of the release callback

### Requirement: Columnar file formats are first-class sources

The adapter SHALL open CSV, Parquet, Arrow IPC/Feather, JSON, and NDJSON files as reopenable tabular
sources with a configurable batch size, an explicit or inferred schema, and projection of the
selected columns. A truncated or malformed file SHALL produce a structured error, never a panic.

#### Scenario: Parquet row groups bound batch size
- **WHEN** a Parquet file is opened with a maximum batch size smaller than a row group
- **THEN** each delivered batch respects the configured size and preserves row order

#### Scenario: Truncated IPC file fails cleanly
- **WHEN** an Arrow IPC stream ends in the middle of a record batch
- **THEN** the source yields a structured read error after the complete batches already delivered

### Requirement: Sciencekit results export to Arrow

The adapter SHALL export dense matrices, CSR matrices, streaming batches, and supervised batches as
Arrow arrays, record batches, or streams with a declared schema, so results can return to the
calling engine without conversion to text.

#### Scenario: Dense predictions become an Arrow column
- **WHEN** a prediction vector is exported
- **THEN** the result is a floating-point Arrow array with the same values and length

#### Scenario: CSR matrix exports with sparse metadata
- **WHEN** a CSR feature block is exported
- **THEN** the export preserves row pointers, column indices, and stored values

### Requirement: Import and export cover both scalar precisions

Every Arrow conversion SHALL support `f32` and `f64` model scalars, converting exactly when the
Arrow dtype matches and by an explicit cast otherwise.

#### Scenario: f32 table with f64 model
- **WHEN** an `f32` Arrow column is selected for an `f64` estimator
- **THEN** values are promoted and the conversion is reported as a cast, not as zero-copy
