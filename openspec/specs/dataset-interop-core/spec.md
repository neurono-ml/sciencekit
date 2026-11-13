# dataset-interop-core Specification

## Purpose
Defines the engine-neutral seam through which heterogeneous, columnar datasets enter sciencekit's
homogeneous dense/sparse and streaming representations, with explicit column roles, missing-value
policy, deterministic ordering, and structured conversion errors.

## Requirements

### Requirement: Tabular schema with named columns and roles

The interop layer SHALL describe an input table as an ordered schema of named columns, each carrying
a logical type and a role of feature, target, or ignored. The feature order SHALL be the declared
selection order, never the source's incidental column order.

#### Scenario: Schema fixes feature order
- **WHEN** a selection lists feature columns `b`, `a` in that order
- **THEN** the materialized feature matrix has `b` as column 0 and `a` as column 1 regardless of the source order

#### Scenario: Duplicate column names are rejected
- **WHEN** two source columns share a name and selection refers to that name
- **THEN** conversion fails with a structured duplicate-column error before any row is read

### Requirement: Explicit feature and target selection

The interop layer SHALL resolve feature columns and at most one target column by name or by
position, and SHALL reject unknown columns, an empty feature selection, and more than one target
column with distinct structured errors.

#### Scenario: Unknown feature column
- **WHEN** a selection names a column absent from the schema
- **THEN** conversion fails with an unknown-column error naming the missing column

#### Scenario: Two targets rejected
- **WHEN** a supervised selection names two target columns
- **THEN** conversion fails before reading rows with an unsupported-operation error

### Requirement: Documented missing-value policy

Every conversion SHALL apply exactly one declared missing-value policy: reject any null, convert
numeric nulls to the floating-point not-a-number sentinel, map nominal nulls to the documented
categorical missing code, or fill with a caller-provided constant. The chosen policy SHALL apply to
every nullable column of the batch.

#### Scenario: Reject policy stops on a null
- **WHEN** a nullable column contains a null and the policy is reject
- **THEN** conversion fails with a structured null-in-column error

#### Scenario: Numeric policy maps null to NaN
- **WHEN** a floating-point column contains a null and the policy is numeric NaN
- **THEN** the materialized block carries the NaN sentinel at that position

#### Scenario: Integer column with null is not silently truncated
- **WHEN** an integer column contains a null under the numeric NaN policy
- **THEN** the column is promoted to the model's floating scalar before the sentinel is written, never truncated

### Requirement: Typed columnar batch materialization

The interop layer SHALL convert a heterogeneous table into a homogeneous numeric feature block by
coercing supported column types to the estimator's scalar type and SHALL reject an unsupported
column type once per batch, before any element is processed.

#### Scenario: Supported numeric columns are coerced
- **WHEN** a batch contains `f32`, `f64`, and `i64` feature columns and the model scalar is `f64`
- **THEN** every value is converted losslessly where representable and the block is dense `f64`

#### Scenario: Unsupported type is rejected once
- **WHEN** a feature column carries a nested or temporal type the numeric path does not support
- **THEN** the batch fails with an unsupported-column-type error naming the column and the suggested conversion

### Requirement: Owned streaming blocks preserve the streaming contract

Converting a tabular batch into a streaming block SHALL produce fully owned data, carrying the
batch position and final indication, so the block remains valid after the source advances. The
conversion SHALL NOT borrow source buffers.

#### Scenario: Block survives source advance
- **WHEN** a tabular batch is converted to a data batch and the source is dropped
- **THEN** the converted block's values and metadata remain valid

#### Scenario: Sparse selection produces a CSR block
- **WHEN** the selected features are declared sparse and the source produces row-aligned values
- **THEN** the streaming block carries a CSR matrix preserving zero entries

### Requirement: Zero-copy only where the representation permits it

The interop layer SHALL document and preserve zero-copy for inputs already stored as a contiguous
row-major matrix of the model scalar, and SHALL copy only when materializing a column-major or
heterogeneous table into the row-major dense view. A zero-copy claim SHALL never be made for a
representation that requires reordering or type coercion.

#### Scenario: Row-major native matrix is borrowed
- **WHEN** an in-memory dense matrix of the model scalar is passed without selection or coercion
- **THEN** the resulting data view borrows the same buffer

#### Scenario: Columnar table is materialized, not aliased
- **WHEN** a multi-column table is converted to a dense row-major view
- **THEN** the conversion allocates a new block and the source buffers are not aliased

### Requirement: Categorical and target extraction through existing views

The interop layer SHALL expose selected nominal or category-coded columns through the existing
categorical view and SHALL expose the selected target column through the existing target view,
delegating label canonicalization to the existing deterministic label table.

#### Scenario: Nominal target is canonicalized deterministically
- **WHEN** the target column contains text labels and is converted for a classifier
- **THEN** the same input sequence yields the same label-to-index table on every run

#### Scenario: Category-coded feature keeps codes
- **WHEN** a feature column is exported with integer category codes
- **THEN** the categorical view carries those codes and documents the missing sentinel

### Requirement: Tabular source contract and provider bridge

The interop layer SHALL define a reopenable, fallible supplier of tabular batches and SHALL provide
an adapter that exposes it as the library's feature-only and supervised source providers, so any
engine adapter reuses one streaming integration.

#### Scenario: Reopenable table source feeds multiple epochs
- **WHEN** a tabular source can be reopened and an iterative estimator requests several passes
- **THEN** each pass receives the table's batches from the beginning in the same order

#### Scenario: Read failure is structured
- **WHEN** the underlying engine fails while producing a batch
- **THEN** the source yields the central taxonomy error and no partial batch is delivered

### Requirement: Async-to-sync bridge with backpressure and cancellation

The interop layer SHALL adapt an asynchronous batch stream to the synchronous source contract on a
dedicated reader thread, using a bounded channel whose depth comes from the execution plan, so the
caller's asynchronous runtime is never blocked and dropping the source cancels the stream.

#### Scenario: Caller runtime is not blocked
- **WHEN** an async engine stream is consumed through the bridge
- **THEN** the caller's thread is never parked on an asynchronous poll

#### Scenario: Dropping the source cancels
- **WHEN** a bridged source is dropped before the stream ends
- **THEN** the reader stops, the underlying stream is dropped, and the thread terminates

### Requirement: Structured interop errors

Interop failures SHALL use the central error taxonomy with distinct variants for unknown or
duplicate columns, unsupported column types, nulls under the reject policy, uneven column lengths,
and schema mismatches, each naming the offending column.

#### Scenario: Uneven columns are rejected
- **WHEN** a table's columns have different lengths
- **THEN** conversion fails with a structured length error before materialization

#### Scenario: Error names the column
- **WHEN** a conversion fails because of a specific column
- **THEN** the error message contains that column's name
