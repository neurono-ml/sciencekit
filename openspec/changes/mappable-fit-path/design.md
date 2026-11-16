# Design

## Context

`SKMappableSource` provides `number_of_rows()` and `row(index)`; the interop crate ships
raw-binary, NumPy and Arrow IPC mapped sources. Estimators that scan rows positionally —
neighbor queries, tree split scans, bootstrap sampling, isolation paths — can therefore
train from disk. The only missing piece is the hand-off: `fit` today takes a borrowed
in-memory view, and no estimator has a mapped entry point.

## Goals / Non-Goals

**Goals**
- One entry point per `RandomAccess` estimator family; no new public trait on the shared
  fit contract (the change is additive).
- Ownership discipline that keeps models safe after the caller drops the source.
- Parity with the in-memory path, proven by the existing acceptance fixtures.

**Non-Goals**
- Streaming (`reopenable-fit-path`), sparse streaming (`sparse-streaming-batches`), GPU.

## Entry point shape

```rust
impl<F> SKKNeighborsClassifier<F> {
    pub fn fit_mappable<S>(&self, features: &S, targets: ...)
        -> Result<SKKNeighborsClassifierModel<F>, SKError>
    where S: SKMappableSource<F>;
}
```

- Free-scope naming rule does not apply: these are methods.
- The shared validation helper `sk_validate_mappable_source` runs before resolution and
  rejects a source without positional access.

## Ownership rule

- **Training-design keepers** (KNN, KNN imputer): the model stores a
  `SKSharedMappableSource<F>` (an `Arc`-backed handle) plus the fitted search structures.
  The mapping stays alive for the model's lifetime; `Send + Sync` is preserved because the
  mapped source already is.
- **Compact-state owners** (trees, forests, bagging, isolation forest, DBSCAN): the model
  owns thresholds, nodes, labels and statistics; the source is dropped after fit.
- **Export:** always materializes owned tensors; import is independent of the file.

## Parity harness

The existing acceptance fixtures (tiny/large/concurrent/deterministic) run twice — once
in memory, once over a mapping — and assert equal predictions within the per-family
tolerance. The harness is a test utility in `sciencekit_common` so every estimator reuses
the same comparison.

## Risks

- **Lifetimes of mapped rows:** avoided by the shared handle; the model never stores a
  borrowed row view.
- **Behavior drift:** the parity harness pins results; a refactor that changes them is a
  bug.
