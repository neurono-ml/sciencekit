# Design

## Context

See `proposal.md` — Why. Rust exposes a JNI cdylib; Arrow C Data Interface / Arrow IPC is the
transport. Spark 4 provides `Dataset.toArrow()` / `DataFrame.toArrow()` (local collection) and
`mapInArrow` for partition-wise processing. Arrow Java can read/write IPC streams in direct
`ByteBuffer`s. Breeze and Smile store data in JVM arrays; ND4J can wrap off-heap direct buffers.

JNI has a hard constraint: crossing the boundary per element is prohibitively slow, and Rust panics
must never unwind into the JVM.

## Goals / Non-Goals

**Goals**

- One Arrow transport for Spark, Arrow vectors, and results.
- Direct-buffer access for ND4J where possible.
- Idiomatic Scala/Java API with typed format options.
- Correct native memory ownership and panic containment.

**Non-Goals**

- No Spark cluster scheduling of sciencekit; only data consumption and optional UDF integration.
- No full support for every Spark SQL type; unsupported types are rejected explicitly.
- No Java implementation of the algorithms; Rust owns computation.
- No JNI callbacks into JVM code from Rust worker threads in the first version.

## Decisions

### Arrow IPC over JNI as the single transport

A native method receives a `ByteBuffer` (direct) containing an Arrow IPC stream and a selection
descriptor (column names/roles/policy), and returns a `ByteBuffer` with the result stream plus a
status code. Strings and errors cross as UTF-8 `jstring`. This avoids per-element JNI and lets
Spark, Arrow Java, and the binding share one path.

Alternative: expose one native method per column type. Rejected because it multiplies the ABI and
still requires a transport for tables.

### Spark: driver vs partition paths

- Driver path: `df.toArrow()` (Spark 4) → Arrow Table → IPC bytes → native. Documented for data that
  fits in driver memory.
- Partition path: `df.mapInArrow(iterator => ..., schema)` (or `mapPartitions` with Arrow writer)
  calls a sciencekit UDF per Arrow batch; the binding provides the batch conversion and a
  `predictInArrow` helper. This is the out-of-core path.

Spark 3.x: no `toArrow()`; the adapter accepts `ColumnarBatch` / Arrow conversion from Spark's
internal Arrow serializer, or falls back to `toPandas` (documented, not the recommended path).

### Smile and Breeze

Smile `DataFrame` exposes `columnNames()` and typed vectors; the adapter reads each selected column
into an Arrow vector builder. Breeze `DenseMatrix` is row-major by default (`DenseMatrix[Double]`
stores rows contiguously), so it can be written to Arrow row-wise or column-wise; the adapter
writes column vectors and lets the Rust side materialize. `CSCMatrix`/`CSRMatrix` map to Arrow
struct columns.

### ND4J

`INDArray.data()` returns a `DataBuffer`; for off-heap or direct buffers,
`data().asNio()`/`pointer()` gives a `ByteBuffer`. The adapter checks `ordering()` (row-major `c` vs
column-major `f`) and `stride()`. Row-major contiguous `FLOAT`/`DOUBLE` is read directly into Arrow
buffers; otherwise a contiguous copy is made and reported.

### Results

Rust returns an Arrow IPC stream; Java builds Arrow vectors, then:
- ND4J: `Nd4j.create(buffer, shape, order)` over a direct buffer where possible.
- Breeze: `DenseMatrix` from the Arrow data.
- Smile: `DataFrame.of(columns)`.
- Spark: `spark.createDataFrame(arrowTable)` or an Arrow-to-Rows iterator.

### Safety

Every JNI entry point wraps the Rust work in `catch_unwind`; a panic becomes a JVM
`IllegalStateException` with the panic message. Native handles are stored in `long` opaque
pointers with explicit `close()`; `Cleaner`/`finalize` is a safety net, not the primary release
path. Arrow `release` callbacks are invoked once.

## Risks / Trade-offs

- **JNI complexity and crash risk** → Mitigation: a tiny ABI (few methods), `catch_unwind`, tests
  that call from many threads, and a memory-leak test using JVM native tracking.
- **Spark version split (3.x vs 4.x)** → Mitigation: compile the Spark module against 4.x; the
  ColumnarBatch path is best-effort and documented; tests run Spark 4 local mode.
- **Duplicate Arrow Java dependencies** → Mitigation: mark Spark/Arrow as `provided` where the host
  supplies them; shade only the binding's own classes.
- **ND4J large dependency** → Mitigation: separate optional module so users without ND4J do not
  pull it.
- **Copy for Smile/Breeze dense data** → Mitigation: document; Arrow transport avoids per-element
  JNI even when a copy is required.

## Migration Plan

New artifacts; additive. Version compatibility table (Java 17+, Scala 2.13/3, Spark 4) documented.

## Open Questions

- Whether to support Spark 3.x `ColumnarBatch` in the first release or document it as unsupported.
  Deferred; the design keeps the hook but tests only Spark 4.
