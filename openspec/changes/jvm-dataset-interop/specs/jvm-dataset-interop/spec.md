# Spec Delta

## Purpose

Defines which JVM table and matrix representations the sciencekit binding accepts and returns — Spark
datasets, Smile and Breeze matrices, ND4J arrays, and Arrow vectors — and how data crosses the JNI
boundary without per-element calls.

## ADDED Requirements

### Requirement: Spark DataFrames and Datasets are accepted

The binding SHALL accept a Spark `Dataset<Row>`/`DataFrame` through Arrow (Spark 4 `toArrow()` or
partition-wise Arrow batches) and SHALL apply feature/target selection by column name.

#### Scenario: Spark frame in local mode
- **WHEN** a Spark `DataFrame` created from a local collection is passed
- **THEN** the binding sees the same rows, columns, and values as the equivalent local data

#### Scenario: Partition-wise out-of-core path
- **WHEN** a Spark frame larger than driver memory is consumed through the partition-wise path
- **THEN** batches are processed per partition without collecting the full frame on the driver

### Requirement: Smile DataFrames are accepted

The binding SHALL accept a Smile `DataFrame`, reading its columns by name and mapping numeric,
boolean, string, and categorical columns through the shared rules.

#### Scenario: Smile numeric frame
- **WHEN** a Smile `DataFrame` with numeric columns is passed
- **THEN** the values and column names match the source

#### Scenario: Smile string column as feature
- **WHEN** a string column is selected as a numeric feature
- **THEN** the binding raises an error suggesting categorical encoding

### Requirement: Breeze dense and sparse matrices are accepted

The binding SHALL accept Breeze `DenseMatrix[Double]`, `DenseVector`, `CSRMatrix`, and `CSCMatrix`;
dense data is copied into the row-major boundary, sparse data converts to CSR preserving stored
zeros, and the conversion path is reported.

#### Scenario: Breeze dense matrix
- **WHEN** a `DenseMatrix[Double]` is passed as features
- **THEN** rows and columns are interpreted correctly and the copy is reported

#### Scenario: Breeze sparse matrix
- **WHEN** a `CSCMatrix` is passed
- **THEN** the binding converts it to CSR with the same logical values and reports the conversion

### Requirement: ND4J arrays are accepted with shape and stride checks

The binding SHALL accept ND4J `INDArray` of `FLOAT` and `DOUBLE`, reading row-major contiguous data
through a direct buffer without per-element JNI calls, and SHALL convert non-contiguous or
column-major arrays explicitly.

#### Scenario: Row-major float array is read directly
- **WHEN** an `INDArray` with row-major order is passed
- **THEN** the binding reads its direct buffer and reports zero per-element copies

#### Scenario: Non-contiguous array is converted
- **WHEN** an `INDArray` with non-standard strides is passed
- **THEN** the binding materializes a contiguous copy and reports it

### Requirement: Data crosses JNI as Arrow buffers

The binding SHALL transport tables and results as Arrow C Data Interface streams or Arrow IPC bytes
in direct `ByteBuffer`s, not as per-element JNI calls, and SHALL release native memory exactly once.

#### Scenario: Large table crosses in batches
- **WHEN** a large Arrow table is passed from the JVM
- **THEN** it is transferred as an Arrow stream in bounded batches

#### Scenario: Native memory is released
- **WHEN** a call completes or fails
- **THEN** the native stream and its buffers are released exactly once

### Requirement: Results return in the requested JVM representation

The binding SHALL return predictions and transforms as ND4J arrays, Breeze matrices, Smile frames,
Spark datasets, or Arrow vectors, selectable by the caller, with values preserved.

#### Scenario: Default prediction type
- **WHEN** `predict` is called with no format option
- **THEN** the result is an ND4J `INDArray` of the model scalar

#### Scenario: Spark result
- **WHEN** a Spark dataset result is requested
- **THEN** the binding creates it from the returned Arrow batches

### Requirement: JVM calls are thread-safe

The binding SHALL allow concurrent calls from multiple JVM threads on independent data with no
shared mutable native state.

#### Scenario: Concurrent fits
- **WHEN** two JVM threads fit separate data concurrently
- **THEN** both produce correct independent models

### Requirement: Errors are structured JVM exceptions

Every conversion failure SHALL raise a JVM exception with the column or array name and a suggested
conversion: `IllegalArgumentException` for unsupported types and shape errors, and a dedicated
`NoSuchElementException` for unknown columns.

#### Scenario: Unknown column
- **WHEN** a feature name absent from the frame is requested
- **THEN** the binding raises `NoSuchElementException` naming it

#### Scenario: Rust panic containment
- **WHEN** a native error occurs
- **THEN** the JNI layer converts it to a JVM exception and does not crash the JVM

### Requirement: Both scalar precisions are supported

Every JVM conversion SHALL support `float` and `double` models, casting explicitly when the element
type differs and reporting the cast.

#### Scenario: Float array with double model
- **WHEN** a `FLOAT` ND4J array is passed to a `double` estimator
- **THEN** values are promoted and the conversion reports a cast
