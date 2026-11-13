# Spec Delta

## Purpose

Defines `SKIncrementalPCA`: covariance accumulation, Chan-Golub-LeVeque merging,
partial/streaming fitting, component and variance state, transform/inverse transform,
representation parity and validation.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

`SKIncrementalPCA` SHALL be constructible only through its builder, exposing
`number_of_components` (default `None`, meaning `min(n_samples, n_features)`), `batch_size`
(default 1024), `whiten` (default false), `maximum_number_of_epochs` (default 1), `backend`
(default `Faer`) and `execution_mode` (default `Automatic`).

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it keeps all feature components, uses batches of 1024, does not whiten, and
  carries `Faer` and `Automatic`

### Requirement: Covariance state and merge

`partial_fit` SHALL update the running sample count, mean and scatter matrix by the
Chan-Golub-LeVeque merge, so a batch is merged into the state without retaining the
previous data.

#### Scenario: Two batches equal one batch

- **WHEN** the same rows are processed as two `partial_fit` calls and as one call
- **THEN** the final mean and scatter matrix agree within the documented tolerance

#### Scenario: State persists across calls

- **WHEN** `partial_fit` is called repeatedly
- **THEN** every call starts from the prior count, mean and scatter

### Requirement: Finalization

The finalize step SHALL derive the covariance from the accumulated scatter, run the
symmetric eigensolver and retain the largest eigenvalues and eigenvectors as components.

#### Scenario: Finalized components are sorted

- **WHEN** a fit completes
- **THEN** explained variance is non-increasing and components correspond to the sorted
  eigenvectors

#### Scenario: Incremental equals full PCA

- **WHEN** the same data is fitted with `SKIncrementalPCA` and `SKPCA`
- **THEN** explained variance and components agree within the documented tolerance

### Requirement: Partial and streaming fitting

`fit` SHALL chunk the input into `batch_size` blocks and call the same update; `fit_streaming`
SHALL drive an `SKSourceProvider` through the shared prefetch executor, supporting multiple
epochs for reopenable providers.

#### Scenario: One-pass streaming

- **WHEN** a non-reopenable provider is used
- **THEN** one pass is consumed and the model is finalized

#### Scenario: Multi-epoch streaming

- **WHEN** a reopenable provider is used with multiple epochs
- **THEN** the source is reopened per epoch and the model records the epochs consumed

#### Scenario: Batch-size invariance

- **WHEN** the same data is fitted with different `batch_size` values
- **THEN** the final components agree within the documented tolerance

### Requirement: Transform, inverse transform and whitening

The model SHALL project a row onto the components, reconstruct an approximation and, when
`whiten` is enabled, scale the projection by the inverse singular-value spread.

#### Scenario: Transform shape

- **WHEN** `m` rows are transformed on a `k`-component model
- **THEN** the result is `m × k`

#### Scenario: Whitening normalizes variance

- **WHEN** whitening is enabled
- **THEN** each transformed component has unit variance within the documented tolerance

### Requirement: Representation parity

`partial_fit`, `fit`, `fit_streaming`, transform and inverse transform SHALL accept dense
and sparse data, dispatch once and never densify a sparse batch.

#### Scenario: Sparse equals dense

- **WHEN** the same logical batches are supplied densely and as CSR
- **THEN** the finalized components agree within the documented tolerance

### Requirement: Validation

The estimator SHALL reject an empty batch, zero or excessive components, a batch smaller
than the component count, non-finite values and a feature-count mismatch across batches.

#### Scenario: Small batch is rejected

- **WHEN** a batch has fewer rows than the component count
- **THEN** `partial_fit` returns `InvalidHyperparameter`

#### Scenario: Feature-count mismatch is rejected

- **WHEN** a later batch has a different feature count from the first
- **THEN** `partial_fit` returns `ShapeMismatch`

### Requirement: Determinism and concurrency

A fixed configuration and input SHALL produce identical models across repeated runs and
thread counts, and a fitted model SHALL be safely shareable.

#### Scenario: Repeated fits agree

- **WHEN** the same estimator fits the same data twice
- **THEN** components and transformed values are identical

#### Scenario: Concurrent transform agrees

- **WHEN** several threads transform with the same fitted model
- **THEN** every result equals the sequential result
