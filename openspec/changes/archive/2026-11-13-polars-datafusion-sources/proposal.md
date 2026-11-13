# Proposal

## Why

The PRD promises `Into<>` conversions from Polars DataFrames/LazyFrames, DataFusion
DataFrames and Arrow/Parquet. Users currently have to materialize and reshape data by hand
before fitting. Feature-gated source adapters let the library consume the Rust data
ecosystem without adding heavy dependencies to the default build.

## What Changes

- Add a tabular source contract to `sciencekit_interop`: schema, column kinds (numeric,
  categorical, boolean, ignored), a dense feature block and an optional categorical view.
- Add fallible conversions from Polars `DataFrame` and `LazyFrame`, DataFusion `DataFrame`
  and `RecordBatch`, and Arrow/Parquet through DataFusion/Polars.
- Column selection, numeric dtype mapping (`f32`/`f64`), categorical/string mapping and
  missing-value mapping (NaN for numeric, the categorical missing sentinel for text).
- Streaming: LazyFrame/DataFusion batches can feed `SKSourceProvider` without collecting
  the whole dataset; materialized conversions produce an owned table.
- Optional features `polars`, `datafusion`, `arrow`, `parquet`; default build stays light.
- Version pins for MSRV 1.85: `datafusion` 46 (MSRV 1.82) or older; `polars` pinned to a
  version verified by the spike (0.46/0.48 candidates); `arrow`/`parquet` aligned with
  DataFusion.
- Validation: unsupported column types, missing columns, shape/row mismatch, non-finite
  numeric values and empty input.
- Explicit non-goals: writing back to Polars/DataFusion, SQL execution beyond DataFrame
  materialization, GPU and Python.
- Companion tests: feature-gated numeric/categorical/missing fixtures, batch iteration,
  schema validation, large data and concurrency.

## Capabilities

### New Capabilities

- `tabular-sources`: the tabular schema/source contract and the Polars/DataFusion/Arrow/
  Parquet conversions, including missing-value and categorical mapping.

### Modified Capabilities

None.

## Impact

- **Code:** `sciencekit_interop` gains a `tabular` module and feature-gated adapters.
- **APIs:** additive.
- **Dependencies:** optional `polars`, `datafusion`, `arrow`, `parquet` with pinned MSRV-
  compatible versions.
- **Downstream:** lets data-engineering workflows feed sciencekit directly and exercises
  the streaming source contract.
- **Acceptance criteria:** feature-gated tests for each integration, large data, schema
  validation, missing/categorical mapping and concurrent reads.
