# truncated-svd Specification

## Purpose
Defines `SKTruncatedSVD`: uncentered randomized low-rank factorization, component and
variance state, transform/inverse transform, dense/sparse parity and validation.

## Requirements

### Requirement: Builder-only construction with documented defaults

`SKTruncatedSVD` SHALL be constructible only through its builder, exposing
`number_of_components` (default 2), `algorithm` (`Randomized` default),
`number_of_iterations` (default 5), `tolerance` (default 0.0), `random_seed` (default 0),
`backend` (default `Faer`) and `execution_mode` (default `Automatic`).

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it uses two components, the randomized algorithm, five iterations, tolerance
  zero, `Faer` and `Automatic`

### Requirement: Uncentered factorization

`SKTruncatedSVD` SHALL factorize the input without subtracting the feature means, and SHALL
store the right singular vectors as components and the singular values of the retained
rank.

#### Scenario: No centering

- **WHEN** a matrix with non-zero feature means is factorized
- **THEN** the reconstruction uses the original means and the model stores no centering
  vector

#### Scenario: Rank-k reconstruction

- **WHEN** a matrix of rank `r` is factorized with `k >= r`
- **THEN** the inverse transform reconstructs the input within the documented tolerance

### Requirement: Dense and sparse parity

Fit, transform and inverse transform SHALL accept dense and sparse views, dispatch once per
operation and never densify a sparse input.

#### Scenario: Sparse equals dense

- **WHEN** the same logical matrix is fitted densely and as CSR with the same seed
- **THEN** components, singular values and transformed values agree within the documented
  tolerance

#### Scenario: Sparse input is not densified

- **WHEN** a wide sparse matrix is fitted or transformed
- **THEN** no dense copy of the input is created

### Requirement: Transform and inverse transform

The model SHALL project a row onto the components and reconstruct an approximation from the
projection.

#### Scenario: Transform shape

- **WHEN** `m` rows are transformed on a `k`-component model
- **THEN** the result is `m × k`

#### Scenario: Inverse transform shape

- **WHEN** `m` projected rows are reconstructed
- **THEN** the result has `m` rows and the fitted feature count

### Requirement: Model state

The model SHALL expose components, singular values, explained variance and the
explained-variance ratio, with the variance computed from the transformed training data.

#### Scenario: Explained variance is non-increasing

- **WHEN** a fit completes
- **THEN** the explained variance sequence is non-increasing and the ratio sums to at most
  one

### Requirement: Validation

Fit SHALL reject empty input, zero or excessive components, negative tolerance,
non-positive iterations, an unsupported algorithm and feature-count mismatch.

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
