# Design

## Context

See `proposal.md` — *Why*. Current constraints:

- `SKDataView<'a, F>` already models `Dense(ArrayView2)` and `Sparse(CsMatView)` and is the
  boundary every in-memory `fit` dispatches on.
- `SKDataBatch` and `SKSupervisedBatch` own an `Array2<F>` (the latter also owns the
  targets), carry a position and a final-block flag, and are moved across threads by the
  prefetch driver.
- `sprs` is already a dependency of `sciencekit_common`; no new crate is needed.
- The execution plan carries the operation's available memory; the provider entry points
  check every batch they read against it.
- `SKOneHotEncoder` streams categorical views, not numeric blocks.

## Goals / Non-Goals

**Goals:**

- One owned block type that both streaming families carry.
- Zero conversion in the driver; the estimator dispatches once, exactly as the in-memory
  `fit` does.
- Honest memory accounting for sparse blocks.

**Non-Goals:**

- Sparse *sources* (Parquet/Arrow/CSR files); this change defines the blocks and the
  estimator consumption, and the interop layer builds the blocks later.
- Sparse streaming for categorical data (one-hot).
- Changing `SKDataView` or the in-memory boundary.

## Decisions

### 1. `SKFeatureBlock<F>` is an owned enum mirroring `SKDataView`

```rust
#[non_exhaustive]
pub enum SKFeatureBlock<F> {
    Dense(Array2<F>),
    Sparse(CsMat<F>),
}
```

with `view() -> SKDataView<'_, F>`, `number_of_rows`, `number_of_columns`, `is_sparse` and
`byte_size`. A borrowed enum already exists for views; the owned twin keeps the same two
variants so dispatch code reads identically in both regimes. Alternatives: a generic
`B: Into<SKDataView>` parameter on the batch — infects every estimator signature and the
prefetch driver with a second type parameter; a trait object — loses the concrete error and
allocation-free matching.

### 2. Batch constructors and accessors

- `SKDataBatch::new_dense(Array2, position, is_final)` and
  `SKDataBatch::new_sparse(CsMat, position, is_final)`; `new(Array2, …)` stays as a dense
  alias for compatibility.
- `data()` is replaced by `view()`, `number_of_rows()`, `number_of_columns()`, `is_sparse()`.
  This is the one breaking change; it is documented in `CHANGELOG.md`. Keeping `data()`
  returning a dense view would silently fail or panic on a sparse block, which is worse than
  a compile error.
- `SKSupervisedBatch` mirrors the same surface and pairs the block with its targets.

### 3. The driver is representation-blind

The prefetch driver moves the block to the compute thread and yields it to the update
closure; it never inspects the representation. Estimators call `view()` and match once, as
their in-memory `fit` already does. This keeps one dispatch point per operation and makes the
sparse path shareable with the dense one.

### 4. Memory accounting follows the stored arrays

`byte_size()` returns:

- dense: `rows · columns · size_of::<F>()`;
- sparse: `data.len() · size_of::<F>() + indices.len() · size_of::<usize>() +
  indptr.len() · size_of::<usize>()`.

The declaration order of `sprs::CsMat` is irrelevant to the accounting; the actual buffers are
counted. The guard keeps its contract (twice the batch size against available memory) but
runs when each batch is read, before the update closure, and receives an honest sparse number
instead of an `n × p` dense one. A byte-size hint still lets resolution reject an obviously
oversized plan early; the read-time check is what every streamed batch goes through.

### 5. Estimators reuse their sparse in-memory paths

- Linear models: the sparse `fit` already iterates CSR rows and computes Gram/retained-design
  state; the streaming update calls the same code per batch. No new algebra.
- Scalers: per-column `count`, `mean`, `M2` (or `min`/`max`, `t-digest`) accumulate over CSR
  rows in `O(nnz)`. Implicit zeros participate as zeros when the statistic requires it
  (mean, moments); `min`/`max` start at zero unless a column is explicitly dense in every
  batch, which the accumulator tracks with a per-column explicit-count.
- Polynomial features: emit a sparse block when the input is sparse, preserving the
  documented column order.
- Imputer: the marker semantics already special-case the CSR implicit zeros for a `0.0`
  marker; the streaming path reuses that predicate per batch.

### 6. One-hot is out of scope

`SKOneHotEncoder::fit_streaming` consumes categorical views. Unifying it requires an owned
categorical block (text columns, integer codes, positions, final flag), which is a separate
container design. It stays on its iterator seam, documented in the book and the specs.

### 7. Sources construct blocks; the driver never converts

A source that reads CSR data builds `CsMat` once per batch and hands it over. If a source has
dense data it builds a dense block. Conversion (`to_dense`, `to_csr`) is always the caller's
choice, never implicit — the same rule the in-memory boundary follows.

## Risks / Trade-offs

- **Implicit zeros distort statistics** → the accumulator tracks per-column explicit counts;
  tests compare each statistic against the dense equivalent on a matrix with many zeros.
- **Hidden densification** → the conformance suite asserts no `n × p` dense allocation for a
  sparse stream (a sentinel test on the block size and a debug counter).
- **Breaking accessors** → documented, migrated across the crate, and caught by the compiler
  rather than at runtime.
- **Memory guard underestimation** → `byte_size` counts the actual buffers, with a test on a
  known CSR layout.
- **Estimator coverage drift** → one shared conformance harness parameterised by estimator
  keeps sparse, dense and streaming agreement asserted in one place.

## Migration Plan

1. Ship `SKFeatureBlock` and the batch changes with the dense path intact.
2. Update the prefetch driver and run the existing suite (dense only) — it must stay green.
3. Migrate estimators one crate at a time, each with its sparse conformance test.
4. Update `CHANGELOG.md` under **Changed** for the breaking accessors.

Rollback is reverting the batch types; no persisted model format changes.
