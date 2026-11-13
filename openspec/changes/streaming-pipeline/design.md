# Design

## Context

See `proposal.md` — Why. `sciencekit_common` supplies `SKLazySource`, `SKSourceProvider`,
`SKFeatureBlock` and the prefetch driver. `type-safe-pipeline` supplies the block medium
and the linear stage contract. The streaming pipeline reuses the executor rather than
inventing a second I/O loop.

## Goals / Non-Goals

**Goals:**
- Reuse the shared prefetch executor and owned batches.
- Fit streaming transformers in one or more epochs without retaining the data.
- A bounded-memory finalizer for batch estimators.

**Non-Goals:**
- `memmap2`-backed batches (Wave 7).
- Streaming DAGs, GPU and export.
- A true out-of-core estimator finalizer (future change).

## Decisions

### Decision 1 — Streaming transformer contract

```rust
pub trait SKStreamingTransformer<F: SKFloat> {
    type Model;
    type Error: From<SKError>;
    fn fit_streaming<S: SKSourceProvider<F>>(&self, source: &mut S, epochs: usize) -> Result<Self::Model, Self::Error>;
    fn transform_batch(&self, model: &Self::Model, batch: &SKFeatureBlock<F>) -> Result<SKFeatureBlock<F>, Self::Error>;
}
```

The fit implementation drives `sk_run_streaming_driver`; the model owns only learned
state, never the source data.

### Decision 2 — Chained fit

Stages are fitted in order. Each stage needs a pass over the source; a reopenable provider
opens a fresh source per pass, a one-shot provider supports only one stage unless the
pipeline materializes the intermediate within the memory budget. The builder exposes
`memory_budget_bytes`; exceeding it returns `BatchBufferOversize` or a shape error rather
than OOM.

### Decision 3 — Per-batch transform

After fitting, the pipeline transforms a batch through every fitted stage in order. Each
stage receives the previous stage's `SKFeatureBlock`; dense/sparse is preserved and no
stage densifies a sparse batch it accepts.

### Decision 4 — Finalizer

The batch finalizer accumulates transformed blocks into an owned `SKFeatureBlock` (rows
concatenated) within the budget, then fits a `SKSupervisedFit`/`SKUnsupervisedFit`
estimator. A streaming estimator finalizer (`partial_fit`) is a documented future
extension; requesting it returns `UnsupportedOperation`.

### Decision 5 — Epochs and errors

`epochs` is validated (≥1); a one-shot provider with epochs >1 returns a structured error.
Source and stage errors propagate through the central taxonomy; state is not rolled back.

### Decision 6 — File layout

```
crates/sciencekit_pipeline/src/streaming/
  mod.rs, transformer.rs, builder.rs, pipeline.rs, finalizer.rs, memory.rs, *_tests.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Chained fitting requires reopening the source** → explicit; the one-shot case is
  documented.
- **Finalizer materializes the transformed data** → bounded by `memory_budget_bytes`; a
  true out-of-core finalizer is future work.
- **Stage state is not rolled back on error** → standard streaming semantics; documented.
- **No `memmap2`** → Wave 7.

## Migration Plan

Additive: new streaming module in `sciencekit_pipeline`. No existing API changes.

## Open Questions

None blocking. A `partial_fit` finalizer and memory-mapped batches are future changes.
