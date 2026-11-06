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

Each integration implements `TryFrom<&DataFrame>` / `TryFrom<RecordBatch>` for
`SKTabularSource<F>` (and `f64`/`f32` variants). The conversion validates column selection,
dtypes, null policy and row counts. A `SKTabularSourceConfig` selects columns and kinds.

### Decision 3 — Streaming adapters

Polars `LazyFrame` and DataFusion `DataFrame` expose a batch provider through the existing
`SKSourceProvider`/`SKLazySource` contracts; each batch converts to `SKFeatureBlock`/
categorical block. Materialized conversions collect all batches into one source. The
streaming path never collects the whole dataset.

### Decision 4 — Missing values

Numeric nulls → NaN. Categorical nulls → `SK_CATEGORICAL_MISSING_CODE`. Nulls in ignored
columns are fine; nulls in unsupported selected columns return `UnsupportedRepresentation`.
Non-finite numeric values outside NaN follow the normal estimator policy.

### Decision 5 — Version pins and features

Features: `polars`, `datafusion`, `arrow`, `parquet`. The spike pins:
- `datafusion` 46 (MSRV 1.82) or 44 (MSRV 1.80.1) if 46's transitive deps exceed 1.85;
- `polars` 0.46/0.48, whichever builds on 1.85;
- `arrow`/`parquet` matching the DataFusion pin.
The default build has no heavy dependency.

### Decision 6 — File layout

```
crates/sciencekit_interop/src/tabular/
  mod.rs, schema.rs, source.rs, categorical.rs, config.rs,
  polars.rs, datafusion.rs, arrow.rs, parquet.rs, *_tests.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

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
