## Why

sciencekit's public boundary is homogeneous: dense row-major matrices, CSR matrices, and owned
streaming blocks. Real datasets arrive as **heterogeneous, columnar tables** — Polars, DataFusion,
Arrow, pandas, Spark — where every column has its own type, nullable entries, and a name. Without
one shared table model, column-selection rules, and missing-value policy, every engine adapter would
reinvent schema mapping and null handling, and the in-memory and out-of-core paths would disagree.

This change builds the neutral seam the concrete adapters (Arrow, Polars, DataFusion, file formats,
memory maps, language bindings) plug into. It adds no engine dependency and no estimator change.

## What Changes

- **New crate `sciencekit_interop`** holding the engine-neutral interop layer: table schema, column
  selection, column data, batch materialization, and source adapters. Concrete engines are optional
  features added by later changes (`arrow`, `polars`, `datafusion`, `memmap`).
- **`SKTabularSchema` / `SKColumnSchema` / `SKColumnRole`**: named columns with logical type and role
  (`Feature`, `Target`, `Ignored`), so an estimator receives a deterministic feature order and at
  most one target column.
- **`SKColumnSelection`**: explicit feature columns and target column by name or position, with
  duplicate/unknown-column rejection before any row is read.
- **`SKMissingValuePolicy`**: `Reject`, `NumericNaN`, `CategoricalMissingCode`, `ConstantFill`, so
  nullable engines have one documented contract instead of ad-hoc null behavior.
- **`SKColumnData` / `SKTabularBatch`**: a column-major batch whose columns are typed (`f32`, `f64`,
  `i64`, `bool`, nominal text, category codes) for the width of one streaming block.
- **Batch conversion methods**: `SKTabularBatch` materializes an owned `SKDataBatch`,
  `SKSupervisedBatch`, `SKDataView` (dense/CSR), `SKTargetView`, and `SKCategoricalView` under the
  selected policy; unsupported types are rejected once, before elements are processed.
- **`SKTabularSource` contract**: a reopenable, fallible supplier of `SKTabularBatch` values that
  adapters implement, plus a bridge into the existing `SKSourceProvider` /
  `SKSupervisedSourceProvider` seams.
- **Async-to-sync bridge**: a dedicated reader thread with a current-thread Tokio runtime and a
  bounded channel, so async engines (DataFusion, Polars `collect_batches`) can feed the sync
  streaming driver without blocking the caller's asynchronous runtime.
- **Structured interop errors** in the central taxonomy: unknown column, duplicate selection,
  unsupported column type, null under `Reject`, uneven column lengths, schema mismatch.
- **Tests**: tiny and large synthetic tables, `f32`/`f64`, nullable and categorical columns,
  concurrency, and error paths; companion `*_tests.rs` modules.
- **Documentation**: a `Development Explained` chapter on the tabular seam and an interop chapter in
  the book.

## Capabilities

### New Capabilities

- `dataset-interop-core`: the engine-neutral tabular seam — schema, column selection, missing-value
  policy, typed column batches, conversion to the library's dense/sparse/streaming views, the
  tabular source contract, the async-to-sync bridge, and the interop error taxonomy.

### Modified Capabilities

<!-- none: the existing data-view-boundary and streaming-batches contracts are consumed, not changed -->

## Impact

- **New crate:** `crates/sciencekit_interop` (folder modules per the 200-line rule).
- **Modified code:** workspace `Cargo.toml` members; the crate may depend only on `ndarray`, `sprs`,
  `thiserror`, `tracing`, `tokio` (bridge only), `num-traits`.
- **Public API:** new types and traits; no existing estimator signature changes.
- **Dependencies:** `tokio` for the bridge (I/O only); engine crates are added by later changes.
- **Sequencing:** first of the interoperability changes; every concrete adapter depends on it.
- **Acceptance (PRD §8.7/§10.3):** conversions verified with little and lots of data, under
  concurrency, for `f32` and `f64`; errors covered by companion `*_tests.rs` modules.
