# Design

## Context

See `proposal.md` — Why. `interop-foundation` provides the neutral batch/source seam and the
async-to-sync bridge; `arrow-interop` provides Arrow import and the C Data Interface. Polars uses
Arrow as its memory model but exposes its own `DataFrame`, `LazyFrame`, `Series`, `ChunkedArray`,
and streaming engine.

Polars' Rust API surface relevant here: `DataFrame` columns are `Series`; a series may have several
chunks; `ChunkedArray::to_ndarray` returns a zero-copy view only for a single chunk with no nulls;
`LazyFrame::collect_batches(engine, maintain_order, chunk_size, lazy)` returns an async
`CollectBatches` (requires the `async` feature) in current Polars versions, and `collect()` /
streaming engine produce a materialized `DataFrame` otherwise.

## Goals / Non-Goals

**Goals**

- Polars users pass `DataFrame`/`LazyFrame` directly; no manual `ndarray` conversion.
- True out-of-core: lazy frames execute in batches, never fully materialized.
- Reuse Arrow for the actual column conversion where possible.

**Non-Goals**

- No Polars expression authoring or query building on sciencekit's side.
- No nested/list/struct columns, no temporal feature expansion.
- No Polars Cloud or remote execution.
- No dependency on Polars' Python package; that belongs to `python-dataset-interop`.

## Decisions

### Eager path reuses the Arrow adapter

`DataFrame` → `RecordBatch`/`Table` via Polars' Arrow export, then the Arrow import path. This
keeps one implementation of null, dictionary, and dtype mapping. For a single-column model scalar
series, use `ChunkedArray::to_ndarray` for a direct zero-copy view instead of a round-trip through
Arrow.

Alternative: convert Polars columns directly to `ndarray` column by column. Rejected because it
duplicates the Arrow conversion and loses dictionary/label semantics.

### Lazy path goes through the async bridge

`LazyFrame::collect_batches(Engine::Streaming, maintain_order = true, chunk_size = Some(n), lazy =
true)` produces an async stream of `DataFrame`s. The foundation's async bridge owns the runtime
thread and exposes a sync `SKTabularSource`. Each collected `DataFrame` goes through the eager
conversion. Reopening clones the `LazyFrame` and starts a new collection.

`maintain_order = true` is mandatory: the streaming contract requires source-order delivery, and
Polars' parallel execution can otherwise reorder batches.

### Projection pushdown

Before collection, apply `LazyFrame::select` with the selected feature and target columns in the
declared order (or `select` the target and features). This makes the unselected columns
unmaterialized. For positional selection, resolve names from `collect_schema()` first.

### Dtype mapping

| Polars dtype | Neutral column |
|---|---|
| `Float32` | `NumericF32` |
| `Float64` | `NumericF64` |
| `Int64` | `Integer` |
| `Boolean` | `Boolean` |
| `String`, `Categorical`, `Enum` | nominal / codes + label table |
| `Int32`, `Int16`, `UInt*` | cast to `Int64` if lossless, else reject |
| everything else | reject with suggestion |

Nulls follow `SKMissingValuePolicy`; the default is reject.

### Export

`Array2<F>` → `DataFrame::new([Series::from_vec(...)])` per column (requires a copy per column, but
the destination is columnar). `CsMat<F>` → three integer/float series plus shape metadata or a
`DataFrame` of `(row, column, value)` triples, with the triple form as the portable default.
Streaming batches export directly through Arrow.

## Risks / Trade-offs

- **Polars version and Arrow version drift** → Mitigation: use the C Data Interface when versions
  differ; pin a supported Polars range in the feature.
- **`collect_batches` API instability** → Mitigation: isolate it behind one adapter module; fall
  back to `collect()` + `iter_chunks` for eager frames.
- **Streaming engine memory spikes** → Mitigation: enforce `chunk_size`; tests assert resident
  batches under the plan's buffer depth.
- **Categorical ordering instability** → Mitigation: use the category mapping from the frame's
  categorical order (which Polars preserves) and test round-trips.
- **Projection changing schema** → Mitigation: resolve names from the pre-projection schema and
  assert the post-projection order.

## Migration Plan

Additive and feature-gated. No existing code changes.

## Open Questions

- Whether to support Polars' new streaming engine-only dtypes in the first iteration. Deferred; the
  adapter rejects unknown dtypes explicitly.
