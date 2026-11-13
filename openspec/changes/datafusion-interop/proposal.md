## Why

DataFusion is the reference SQL/DataFrame query engine written in Rust. It plans, optimizes, and
executes queries over Arrow data and can push filters and projections into Parquet, CSV, JSON, and
object stores. Sciencekit should be able to consume a DataFusion `DataFrame` as a streaming source
so users keep DataFusion's pushdown and get out-of-core fitting, instead of exporting results and
loading them back.

DataFusion is asynchronous, while sciencekit's streaming source contract is synchronous. This change
supplies the adapter and the async bridge usage that keeps the async runtime unblocked.

## What Changes

- **DataFusion feature on `sciencekit_interop`** (`datafusion`) with the Arrow feature.
- **`DataFrame` import**: a DataFusion `DataFrame` becomes a reopenable `SKTabularSource` through
  `execute_stream`, with projection and selection translated before execution.
- **Async bridge**: the `SendableRecordBatchStream` is polled on the foundation's dedicated reader
  thread; batches are delivered in order with bounded buffering and cancellation on drop.
- **RecordBatch conversion**: DataFusion batches use the Arrow adapter, so null, dictionary, and
  dtype semantics are shared.
- **Export/registration**: sciencekit blocks can be registered as a DataFusion `MemTable` and
  exposed as a `DataFrame`, for pipelines that mix sciencekit transforms with SQL.
- **Tests**: SQL pushdown, multiple partitions, ordered delivery, cancellation, error propagation,
  small and large datasets, concurrency, `f32`/`f64`.

## Capabilities

### New Capabilities

- `datafusion-source-interop`: DataFusion `DataFrame` and record-batch streams as sciencekit
  streaming sources, with projection pushdown, ordered async bridging, cancellation, and result
  registration back into a DataFusion session.

### Modified Capabilities

<!-- none -->

## Impact

- **Dependencies:** `datafusion`, `tokio` (already in the foundation bridge), and Arrow crates
  behind the `datafusion` feature.
- **New code:** `crates/sciencekit_interop/src/datafusion/` (`mod`, `source`, `bridge`,
  `registration`, companion `*_tests.rs`).
- **Public API:** added behind the feature.
- **Sequencing:** depends on `interop-foundation` and `arrow-interop`.
- **Acceptance (PRD §8.7/§10.3):** streamed fit equals dense fit; multiple partitions delivered in
  deterministic order; calls do not block the caller's Tokio runtime; large and small data;
  `f32`/`f64`; model export and metrics unaffected.
