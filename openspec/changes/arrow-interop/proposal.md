## Why

Apache Arrow is the shared memory model behind Polars, DataFusion, pandas/pyarrow, Spark, Julia's
`Arrow.jl`, and most columnar file formats. Making Arrow the first concrete adapter of the
neutral seam delivers four things at once: a zero-copy in-process exchange format, the C Data
Interface that every language binding can speak, a streaming `RecordBatchReader`, and readers for
CSV, Parquet, Arrow IPC/Feather, JSON, and NDJSON.

Without this adapter, each engine and each binding would parse its own files and invent its own
column conversion.

## What Changes

- **Arrow feature on `sciencekit_interop`** (`arrow`) depending on `arrow-array`, `arrow-schema`,
  `arrow-buffer`, `arrow-cast`, `arrow-select`, `arrow-ipc`, `arrow-csv`, `arrow-json`, and
  `parquet` (Arrow-enabled).
- **In-memory import**: `RecordBatch`, `Table`, `Array`, `ChunkedArray`, `RecordBatchReader` →
  `SKTabularBatch`, with primitive-array zero-copy where the layout permits it.
- **Streaming import**: `RecordBatchReader` and `FFI_ArrowArrayStream` → reopenable
  `SKTabularSource`, with bounded batch size and projection pushdown (only selected columns read).
- **C Data Interface**: export `ArrowSchema` / `ArrowArray` / `ArrowArrayStream` (and import them)
  with correct ownership and release callbacks, so Python, Julia, and JVM bindings can receive
  sciencekit results and send tables without an intermediate format.
- **Null handling**: Arrow validity bitmaps feed the missing-value policy from
  `dataset-interop-core`; dictionary arrays become categorical columns; string arrays become
  nominal columns; boolean arrays become categorical or numeric according to the column role.
- **File formats**: path-based, reopenable sources for CSV, Parquet, Arrow IPC/Feather, JSON, and
  NDJSON. Parquet supports row-group batching and column projection; CSV/JSON support explicit or
  inferred schema; all use a configurable batch size.
- **Export**: `Array2`, `CsMat`, `SKDataBatch`, and `SKSupervisedBatch` → Arrow arrays/record
  batches/streams, for returning predictions and transformed blocks.
- **Tests**: round-trip Arrow import/export, null and dictionary handling, projection, batch sizes,
  file formats with small and large files, concurrency, `f32`/`f64`.
- **Documentation**: an Arrow interop chapter and a `Development Explained` chapter for the C Data
  Interface and file readers.

## Capabilities

### New Capabilities

- `arrow-columnar-interop`: Arrow in-memory and streaming import/export, the C Data Interface,
  null/dictionary/string mapping, and file-format sources.

### Modified Capabilities

<!-- none -->

## Impact

- **Dependencies:** Arrow crates and `parquet` behind the `arrow` feature; the core crate stays
  dependency-light when the feature is off.
- **New code:** `crates/sciencekit_interop/src/arrow/` (`mod`, `import`, `export`, `ffi`,
  `reader_sources`, `file_sources`, `formats`, plus companion `*_tests.rs`).
- **Public API:** added behind the feature; no existing signature changes.
- **Sequencing:** depends on `interop-foundation`; unlocks `polars-interop`,
  `datafusion-interop`, and every language-binding change.
- **Acceptance (PRD §8.7/§10.3):** small and large tables, `f32`/`f64`, concurrent readers,
  round-trip equality within floating-point tolerance, and model export/metrics unaffected.
