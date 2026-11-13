# Design

## Context

See `proposal.md` — Why. `sciencekit_common` owns `SKFeatureBlock`, `SKNominalMatrixView`
and the source contracts. `sciencekit_interop` owns format adapters. MSRV findings:
`datafusion 55` requires Rust 1.94, so the plan pins `datafusion 46` (MSRV 1.82) or older;
`polars` declares no rust-version, so 0.46/0.48 are spike candidates; `arrow`/`parquet`
must match the DataFusion pin.

## Goals / Non-Goals

**Goals:**
- A small, schema-aware tabular source shared by all integrations.
- Feature-gated adapters that keep the default build light.
- Correct missing/categorical mapping.

**Non-Goals:**
- Writing back to Polars/DataFusion.
- SQL execution beyond reading a DataFrame.
- GPU and Python.

## Decisions

### Decision 1 — `SKTabularSource` with schema + numeric + categorical

```rust
pub struct SKTabularSource<F: SKFloat> {
    pub schema: SKTabularSchema,
    pub features: SKFeatureBlock<F>,
    pub categorical: Option<SKNominalMatrixOwned<F>>,
}
```

`SKTabularSchema` lists column names and kinds. Numeric columns form a dense block;
categorical/string columns form an owned nominal matrix (codes + values + missing
sentinel). Boolean columns map to 0/1 numeric columns. Ignored columns are dropped.

### Decision 2 — Fallible conversions

Each integration exposes a free conversion function (and, where a materialized value makes
sense, a `TryFrom<(&DataFrame, &SKTabularSourceConfig)>`) for `SKTabularSource<F>` (and
`f64`/`f32` variants). The conversion validates column selection, dtypes, null policy and row
counts. A `SKTabularSourceConfig` selects columns and kinds.

### Decision 3 — Streaming adapters

Polars `LazyFrame` and DataFusion `DataFrame` expose a batch provider through the existing
`SKSourceProvider`/`SKLazySource` contracts; each batch converts to `SKFeatureBlock`/
categorical block. Materialized conversions collect all batches into one source.

DataFusion runs the physical plan through `execute_stream` and never collects the whole result
set. Polars 0.46 exposes no incremental stream API over a `LazyFrame` (only `collect` and
`sink_*`), so `SKPolarsLazySource` executes the plan once per `open` and slices the executed
frame into owned batches of at most `batch_size` rows. That keeps downstream memory bounded
per block and row order intact, but the plan itself is not pulled row by row; the type and the
mdBook chapter state this limitation instead of claiming streaming the engine cannot do.

### Decision 4 — Missing values

Numeric nulls → NaN. Categorical nulls → `SK_CATEGORICAL_MISSING_CODE`. Nulls in ignored
columns are fine; nulls in unsupported selected columns return `NullValueInColumn` under
the reject policy, and infinite values return `NonFiniteValue`.
Non-finite numeric values outside NaN follow the normal estimator policy.

### Decision 5 — Version pins and features

Features: `polars`, `datafusion`, `arrow`, `parquet`. The spike result:
- `polars` 0.46 with `lazy`, `strings` and `dtype-categorical` builds on Rust
  1.85 **only** with `psm` pinned to 0.1.28: newer `psm` build-dependencies pull
  `ar_archive_writer` 0.5.x, which uses let-chains from Rust 1.88;
- `datafusion` 46 (MSRV 1.82) is the newest release on the toolchain and pins
  Arrow 54.2 internally;
- `arrow`/`parquet` stay at the workspace's 58.3.0; the DataFusion adapter
  translates its own Arrow 54 batches into the neutral column vocabulary rather
  than mixing two Arrow crates in one public type.
The default build has no heavy dependency.

### Naming deviation

The existing `crate::source::SKTabularSource` trait already owns the name of the
reopenable batch supplier. The materialized value described above is therefore
published as `crate::tabular::SKTabularDataset<F>`; it *is* the tabular source
in the data sense, while the trait keeps the streaming sense. The owned
categorical block (`SKNominalMatrixOwned`) is not generic: category codes are
integers regardless of the model scalar.

### Decision 6 — File layout

```
crates/sciencekit_interop/src/
  tabular/
    mod.rs, column_kind.rs, config.rs, dataset.rs, materialization.rs,
    nominal_matrix_owned.rs, *_tests.rs
  polars/
    mod.rs, dtype_mapping.rs, column_conversion.rs, dataframe_conversion.rs,
    lazy_source.rs, polars_tests.rs
  datafusion/
    mod.rs, dtype_mapping.rs, column_conversion.rs, batch_conversion.rs,
    dataframe_conversion.rs, parquet_conversion.rs, stream_source.rs,
    datafusion_tests.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines (the test
companions are the exception).

## Risks / Trade-offs

- **Polars/DataFusion are heavy and their MSRV is not declared** → feature-gated and
  spike-pinned; default build unaffected.
- **Categorical ownership** → an owned nominal block is needed; it is small relative to the
  numeric data.
- **Null semantics differ per backend** → one documented mapping and tests per integration.
- **Streaming batch conversion** → row order preserved; batches are owned.
- **Version drift** → a compatibility table in the crate docs.

## Migration Plan

Additive: feature-gated modules in `sciencekit_interop`. No existing API changes.

## Open Questions

None blocking; the exact Polars version is resolved by the spike.
