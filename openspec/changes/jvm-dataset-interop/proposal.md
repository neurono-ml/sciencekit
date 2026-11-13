## Why

The JVM is where enterprise data lives: Apache Spark's `Dataset`/`DataFrame`, Smile's `DataFrame`,
Breeze's dense/sparse matrices, and ND4J's `INDArray`. Scala users expect to pass those values
directly. A JVM binding that only accepts primitive arrays would force a serialization step on every
call and exclude Spark, the most common source of large datasets.

Spark 4 exposes Arrow directly (`DataFrame.toArrow()`), and Arrow Java can consume the Arrow C Data
Interface / IPC stream, so the same Arrow bridge used by Python and Julia covers the JVM without a
per-engine protocol.

## What Changes

- **New JVM binding**: a `sciencekit-java` artifact with a JNI cdylib and an idiomatic Scala
  wrapper (`sciencekit-scala`), including `ScienceKit` entry points for fit/predict/transform.
- **Spark `Dataset<Row>`/`DataFrame`**: consumed through Spark 4 `toArrow()` or Arrow IPC batches;
  for out-of-core pipelines, a `mapInArrow` / `mapPartitions` integration feeds Arrow batches, with
  the driver-collect path documented as bounded.
- **Smile `DataFrame`**: consumed by reading its column vectors; categorical and string columns map
  to nominal/table data.
- **Breeze**: `DenseMatrix[Double]`, `DenseVector`, `CSCMatrix`, and `CSRMatrix` accepted; dense
  matrices are copied into the row-major boundary, sparse matrices converted to CSR.
- **ND4J `INDArray`**: accepted via its direct `ByteBuffer` for `FLOAT`/`DOUBLE`, preserving shape
  and strides when row-major contiguous; otherwise converted.
- **Arrow IPC over JNI**: results and inputs cross as `ArrowArrayStream`/IPC bytes in direct
  `ByteBuffer`s, avoiding per-element JNI calls.
- **Return values**: JVM results return as `INDArray`, Breeze `DenseMatrix`, Smile `DataFrame`,
  Spark `Dataset` (via `createDataFrame`), or Arrow vectors, selectable by the caller.
- **Errors**: structured Java/Scala exceptions mapping `SKError` variants.
- **Tests**: Spark local mode, Smile, Breeze, ND4J, Arrow round-trips, large and small data,
  concurrency, `float`/`double`.

## Capabilities

### New Capabilities

- `jvm-dataset-interop`: JVM table and matrix representations (Spark, Smile, Breeze, ND4J, Arrow)
  as sciencekit inputs and outputs, with an Arrow-over-JNI transport and structured errors.

### Modified Capabilities

<!-- none -->

## Impact

- **New code:** `crates/sciencekit_jvm_abi` (JNI cdylib) and `java/` (Maven/Gradle multi-module:
  `sciencekit-java`, `sciencekit-scala`).
- **Dependencies (JVM):** Arrow Java, Spark 4 (`spark-sql` provided), Smile, Breeze, ND4J
  (provided/optional per module); JNI crate on the Rust side.
- **Dependencies (Rust):** `sciencekit_interop` with `arrow`, `jni`.
- **Sequencing:** depends on `interop-foundation`, `arrow-interop`, and the binding ABI. The PRD
  currently lists only Python bindings; this change also updates the PRD's binding roadmap.
- **Acceptance (PRD §8.7/§10.3):** every representation fits the same estimator with equal results;
  Spark local mode with multiple partitions; large and small data; `float`/`double`; model export
  and metrics unaffected.
