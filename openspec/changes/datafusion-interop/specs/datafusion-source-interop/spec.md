# Spec Delta

## Purpose

Defines how DataFusion lazy DataFrames and record-batch streams become sciencekit streaming sources
without consuming the whole result, blocking the caller's async runtime, or losing partition order,
and how sciencekit blocks can be registered back into a DataFusion session.

## ADDED Requirements

### Requirement: DataFusion DataFrame becomes a streaming source

The adapter SHALL expose a DataFusion `DataFrame` as a reopenable tabular source by executing it
with `execute_stream`, converting each Arrow record batch through the Arrow adapter, and SHALL NOT
collect the full result into memory.

#### Scenario: Large DataFrame is streamed
- **WHEN** a DataFusion DataFrame over a file larger than the configured memory budget is imported
- **THEN** batches arrive incrementally and resident memory stays bounded by the plan buffer depth

#### Scenario: Reopenable source reruns the query
- **WHEN** an iterative estimator requests a second pass from a reopenable DataFusion source
- **THEN** the query executes again and delivers the same batches in the same order

### Requirement: Async execution never blocks the caller's runtime

The adapter SHALL poll the DataFusion stream on a dedicated reader thread and deliver owned batches
through a bounded channel, so no caller thread waits on an asynchronous poll.

#### Scenario: Caller runtime stays responsive
- **WHEN** a sciencekit fit consumes a DataFusion stream from inside a Tokio runtime
- **THEN** other tasks on that runtime continue to make progress during the fit

#### Scenario: Backpressure follows the plan
- **WHEN** the execution plan resolves a buffer depth of one
- **THEN** at most two batches are resident between the engine and the consumer

### Requirement: Batch order is deterministic

The adapter SHALL deliver batches preserving the query's row order even when execution uses
multiple partitions, and SHALL mark exactly one final batch.

#### Scenario: Multiple partitions preserve row order
- **WHEN** a query executes with four partitions
- **THEN** the concatenated delivered rows equal the single-partition result in order

### Requirement: Selection and filters are pushed down

The adapter SHALL apply the feature/target projection and any caller-supplied filter to the
DataFusion plan before execution, so unselected columns and filtered rows are not read.

#### Scenario: Projection reaches the plan
- **WHEN** three of twelve columns are selected from a Parquet-backed DataFrame
- **THEN** the physical plan reads only those three columns

### Requirement: Cancellation and errors are structured

Dropping the source SHALL cancel execution and release the stream; an execution error SHALL surface
as the central taxonomy error after previously delivered batches, and SHALL never panic.

#### Scenario: Dropping mid-stream cancels
- **WHEN** the consumer drops the source after the first batch
- **THEN** the reader stops polling, the stream is dropped, and the thread terminates

#### Scenario: Query failure is reported
- **WHEN** the underlying scan fails after two batches
- **THEN** the third read returns a structured error and the first two batches remain valid

### Requirement: Sciencekit blocks can be registered in DataFusion

The adapter SHALL expose dense matrices, CSR matrices, and supervised batches as DataFusion tables
whose `DataFrame` reproduces the same rows and columns.

#### Scenario: Dense block becomes a table
- **WHEN** a dense feature block with named columns is registered
- **THEN** a query over it returns the same rows and column names

#### Scenario: Sparse block becomes a table
- **WHEN** a CSR feature block is registered
- **THEN** the query returns the same logical values, with the sparse representation preserved in metadata

### Requirement: Both scalar precisions are supported

The adapter SHALL support `f32` and `f64` model scalars, casting Arrow columns explicitly when the
dtype differs.

#### Scenario: f32 result with f64 model
- **WHEN** a DataFusion query yields `Float32` columns for an `f64` estimator
- **THEN** values are promoted and the conversion reports a cast
