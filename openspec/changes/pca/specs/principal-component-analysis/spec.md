# Spec Delta

## Purpose

Defines `SKPCA`: builder surface, centering, full/randomized/covariance solvers, component
and variance state, transform/inverse transform, whitening, the dense-only exception and
validation.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

`SKPCA` SHALL be constructible only through its builder, exposing `number_of_components`
(default `None`), `svd_solver` (`Auto` default), `whiten` (default false), `tolerance`
(default 0.0), `number_of_power_iterations` (default 5), `random_seed` (default 0),
`backend` (default `Faer`) and `execution_mode` (default `Automatic`).

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it keeps all possible components, resolves the solver automatically, does not
  whiten, uses five power iterations, `Faer` and `Automatic`

#### Scenario: Backend reaches the fitted model

- **WHEN** a non-default backend is configured and the estimator is fitted
- **THEN** the fitted model records that backend in its transform spans

### Requirement: Centering

Fit SHALL compute the feature means and center the design before any solver runs, and the
model SHALL store the mean for transform and inverse transform.

#### Scenario: Centered data has zero mean

- **WHEN** a fitted model's stored mean is subtracted from the training data
- **THEN** every feature has mean zero within the documented tolerance

### Requirement: Full SVD solver

The `Full` solver SHALL compute the full singular-value decomposition of the centered
design and retain the requested number of right singular vectors as components.

#### Scenario: Full solver reconstructs a small design

- **WHEN** a small design is fitted with `Full`
- **THEN** the components and explained variance reconstruct the centered data within
  tolerance

### Requirement: Randomized solver

The `Randomized` solver SHALL use the randomized SVD kernel with the configured power
iterations and seed, and SHALL return components and singular values consistent with the
full solver within the documented tolerance for low-rank data.

#### Scenario: Randomized approximates full

- **WHEN** the same low-rank design is fitted with `Full` and `Randomized`
- **THEN** explained variance and components agree within the documented tolerance

#### Scenario: Randomized is reproducible

- **WHEN** the same seed and data are fitted twice
- **THEN** components and transformed values are identical

### Requirement: Covariance eigensolver

The `CovarianceEigh` solver SHALL form the covariance matrix of the centered design, run a
symmetric eigendecomposition and retain the largest eigenvalues and their eigenvectors as
components.

#### Scenario: Covariance solver equals full SVD

- **WHEN** the same design is fitted with `CovarianceEigh` and `Full`
- **THEN** explained variance and components agree within the documented tolerance

### Requirement: Automatic solver resolution

`Auto` SHALL choose a deterministic solver from the input shape and component count: full
SVD for small designs, randomized SVD for large designs, and covariance eigh when the
number of samples clearly exceeds the number of features. The resolved solver SHALL be
exposed on the model.

#### Scenario: Small design uses full SVD

- **WHEN** a small design is fitted with `Auto`
- **THEN** the model reports `Full`

#### Scenario: Large design uses randomized SVD

- **WHEN** a large design is fitted with a small component count and `Auto`
- **THEN** the model reports `Randomized`

### Requirement: Transform, inverse transform and whitening

The model SHALL project a row onto the components, SHALL reconstruct an approximation from
the projection, and when `whiten` is enabled SHALL scale the projected components by the
inverse singular-value spread.

#### Scenario: Transform shape

- **WHEN** `m` rows are transformed on a `k`-component model
- **THEN** the result is `m × k`

#### Scenario: Inverse transform reconstructs the training data

- **WHEN** the full component count is used without whitening
- **THEN** the inverse transform of the transform reproduces the centered data

#### Scenario: Whitening normalizes component variance

- **WHEN** whitening is enabled
- **THEN** every transformed component has unit variance within the documented tolerance

### Requirement: Model state

The model SHALL expose components, explained variance, explained-variance ratio, singular
values, the mean and the resolved solver.

#### Scenario: Explained variance is sorted

- **WHEN** a fit completes
- **THEN** explained variance is non-increasing and the ratio is the variance divided by
  the total variance

### Requirement: Dense-only exception

PCA SHALL accept dense input only and SHALL reject a sparse view with
`UnsupportedRepresentation` and a suggestion to use `SKTruncatedSVD`, because centering a
sparse matrix destroys its sparsity.

#### Scenario: Sparse input is rejected

- **WHEN** a sparse view is passed to fit or transform
- **THEN** the operation returns `UnsupportedRepresentation` with the alternative suggestion

#### Scenario: Dense input works

- **WHEN** a dense view is fitted and transformed
- **THEN** the operation succeeds through the dense path

### Requirement: Validation

Fit SHALL reject empty input, zero or excessive components, negative tolerance, zero power
iterations, an invalid solver and feature-count mismatch.

#### Scenario: Excessive components are rejected

- **WHEN** `number_of_components` exceeds `min(n_samples, n_features)`
- **THEN** fit returns `InvalidHyperparameter`

#### Scenario: Feature-count mismatch is rejected

- **WHEN** transform receives a different feature count from fit
- **THEN** it returns `ShapeMismatch`

### Requirement: Determinism and concurrency

A fixed configuration and seed SHALL produce identical models and transforms across runs
and thread counts, and a fitted model SHALL be safely shareable.

#### Scenario: Repeated fits agree

- **WHEN** the same estimator fits the same data twice
- **THEN** components and transformed values are identical

#### Scenario: Concurrent transform agrees

- **WHEN** several threads transform with the same fitted model
- **THEN** every result equals the sequential result
