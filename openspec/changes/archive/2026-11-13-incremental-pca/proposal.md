# Proposal

## Why

PCA needs the whole centered design in memory. IncrementalPCA accumulates the covariance
(mean and scatter matrix) batch by batch using the Chan-Golub-LeVeque merge, so it can fit
from `partial_fit` batches or a streaming source without retaining the dataset. It is the
out-of-core decomposition path and the natural consumer of the streaming executor.

## What Changes

- `SKIncrementalPCA<F>` in `sciencekit_decomposition`, with a distinct fitted model.
- Builder-only surface: `number_of_components` (default `None`, meaning the current
  feature count), `batch_size` (default 1024), `whiten` (default false), `backend`
  (default `Faer`) and `execution_mode` (default `Automatic`).
- `partial_fit` merges a batch into the running mean and scatter matrix; `fit` accepts a
  full design; `fit_streaming` drives an `SKSourceProvider` through the shared prefetch
  executor with multi-epoch support for reopenable providers.
- The finalize step computes the covariance, runs the symmetric eigensolver and stores
  components, explained variance, explained-variance ratio and the mean.
- Dense and sparse batches are accepted; the covariance/scatter accumulation uses sparse
  products and never densifies a sparse batch. Transform and inverse transform operate on
  dense or sparse views.
- Validation: empty batch, zero or excessive components, batch smaller than the component
  count, non-finite values and feature-count mismatch across batches.
- Explicit non-goals: missing values, GPU, `memmap2` and export (Wave 7), and the exact
  full-SVD solver (`SKPCA` owns it).
- Companion tests: hand-checkable two-batch merge, equality with PCA on the same data,
  batch-size invariance, streaming equals repeated `partial_fit`, sparse/dense equivalence,
  whitening, determinism, concurrency and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `incremental-pca`: `SKIncrementalPCA` — covariance accumulation, Chan-Golub-LeVeque
  merge, partial/streaming fitting, component/variance state, transform/inverse transform,
  representation parity and validation.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_decomposition` gains the incremental-PCA module and model.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common` (streaming sources and executor), `sciencekit_math`,
  `ndarray`, `sprs`, `rayon`; `sciencekit_metrics` as a dev-dependency.
- **Downstream:** gives pipelines an out-of-core reduction stage and exercises the
  reopenable streaming-provider path for a non-supervised estimator.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent transform,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Export is deferred
  to Wave 7 and recorded in the PRD.
