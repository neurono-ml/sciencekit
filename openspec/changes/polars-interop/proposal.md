## Why

Polars is the most widely used Rust DataFrame library and a first-class citizen in Python. Users
expect to pass a `DataFrame` or `LazyFrame` directly to a model instead of exporting CSV or
converting to `ndarray` by hand. Polars is also built on Arrow, so its in-memory representation can
reuse the Arrow adapter — while its lazy streaming engine can feed sciencekit's out-of-core driver
batch by batch.

This change makes Polars eager and lazy data a native input and output of sciencekit.

## What Changes

- **Polars feature on `sciencekit_interop`** (`polars`) with the Arrow feature.
- **Eager `DataFrame` / `Series` import**: selected columns become a tabular batch; single-chunk
  primitive columns of the model scalar are borrowed, multi-chunk or multi-column data is
  materialized.
- **Lazy `LazyFrame` import**: a `LazyFrame` with a streaming engine and a configured chunk size
  becomes a reopenable `SKTabularSource`, using Polars' `collect_batches` path; batch order and
  final marking follow the streaming contract.
- **Column semantics**: Polars `Float32`/`Float64`/`Int64`/`Boolean`/`String`/`Categorical`/`Enum`
  map to the neutral column types; nulls follow the missing-value policy; all other dtypes are
  rejected with a suggested cast.
- **Selection pushdown**: the feature/target selection is translated into a Polars column selection
  before execution, so unselected columns are never materialized.
- **Export**: `Array2`, `CsMat`, `SKDataBatch`, `SKSupervisedBatch` convert to Polars
  `DataFrame`/`Series`/`LazyFrame`.
- **Tests**: eager and lazy parity, chunked columns, nulls, categoricals, projections, small and
  large frames, concurrency, `f32`/`f64`.

## Capabilities

### New Capabilities

- `polars-source-interop`: Polars eager and lazy tables as sciencekit inputs and outputs, including
  streaming collection, column selection, dtype mapping, null policy, and result export.

### Modified Capabilities

<!-- none -->

## Impact

- **Dependencies:** `polars` (with `lazy`, `streaming`, `ndarray`, `dtype-full` features as
  required) behind the `polars` feature; Polars' Arrow version must match the Arrow adapter or the
  conversion goes through the C Data Interface.
- **New code:** `crates/sciencekit_interop/src/polars/` (`mod`, `eager`, `lazy`, `export`,
  `dtypes`, companion `*_tests.rs`).
- **Public API:** added behind the feature.
- **Sequencing:** depends on `interop-foundation` and `arrow-interop`.
- **Acceptance (PRD §8.7/§10.3):** eager `fit` equals lazy streamed `fit` on the same frame; large
  and small frames; concurrent readers; `f32`/`f64`; model export and metrics unaffected.
