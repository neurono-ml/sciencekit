# Design

## Context

The streaming driver (`sciencekit_common::execution::streaming_driver`) already
orchestrates prefetch and per-pass policies; what is missing is a `fit_streaming`
implementation on the iterative estimators, which need to restart the provider between
passes. `SKReopenableSource` already expresses that capability; the estimators just never
ask for it.

## Goals / Non-Goals

**Goals**
- A uniform multi-pass fit for `Iterative` estimators.
- Deterministic epochs: every pass sees the same logical data.
- One parity harness reused by every estimator.

**Non-Goals**
- Mappable fit, sparse batches, GPU, new estimators.

## Shape

```rust
impl<F> SKKMeans<F> {
    pub fn fit_streaming<S>(&self, provider: &S) -> Result<SKKMeansModel<F>, SKError>
    where S: SKReopenableSource<F>;
}
```

- `sk_validate_reopenable_source` checks the contract before the first pass drains the
  provider.
- Each estimator keeps its in-memory solver; the streaming path loops the provider and
  accumulates through the same kernels (norm expansion, scatter, Gram) the in-memory fit
  uses.

## Per-estimator notes

- **KMeans:** passes over batches accumulate per-center sums and counts; assignment uses
  the previous centers; the convergence test compares against the documented tolerance.
- **PCA / TruncatedSVD:** the randomized solver runs its projection passes over the
  stream; the covariance solver accumulates the scatter like `SKIncrementalPCA`.
- **KNNImputer:** the donor pass reopens the stream per target row range; the memory
  guard sizes the row window.
- **MissingIndicator:** one pass suffices; it shares the streaming vocabulary of the
  scalers.

## Risks

- **Provider reopen cost:** reopening is a cheap reset by contract; a provider that cannot
  reopen is rejected up front.
- **Determinism:** epoch order is fixed by the provider; the parity harness compares the
  final model, not intermediate states.
