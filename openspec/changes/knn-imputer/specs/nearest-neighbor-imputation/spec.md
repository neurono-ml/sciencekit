# Spec Delta

## Purpose

Defines the nearest-neighbor imputer: which cells count as missing, how donors are chosen
with the NaN-aware Euclidean distance, how weights and fallbacks fill a cell, how the
missingness indicator composes, and why the transform is dense-only and memory bounded.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

`SKKNNImputer` SHALL be constructible only through its builder, exposing `missing_values`
(default `NaN`), `nearest_neighbors_count` (default 5), `weights` (`Uniform` default),
`add_indicator` (default false), `keep_empty_features` (default false), `backend` (default
`Faer`) and `execution_mode` (default `Automatic`).

#### Scenario: Default configuration

- **WHEN** an imputer is built without calling any setter
- **THEN** it treats `NaN` as missing, uses five uniform donors, drops all-missing
  columns, stacks no indicator, and carries `Faer` with an `Automatic` intent

#### Scenario: Backend reaches the model

- **WHEN** a non-default backend is configured and the imputer is fitted
- **THEN** the fitted model records that backend in its transform span

### Requirement: Missing-marker semantics

The imputer SHALL treat `NaN` as missing through `is_nan` and any other configured scalar
marker through equality, consistently across fit and transform.

#### Scenario: NaN marker

- **WHEN** `NaN` is the marker
- **THEN** every NaN cell is imputed and finite values are preserved

#### Scenario: Scalar marker

- **WHEN** `-999` is the marker
- **THEN** every `-999` cell is imputed and `NaN` is rejected as a non-finite value

#### Scenario: Sparse input is rejected

- **WHEN** a sparse view is passed to fit or transform
- **THEN** the operation returns `UnsupportedRepresentation` with a suggestion to use
  `SKSimpleImputer`, because a CSR implicit zero is an observed zero

### Requirement: Fit learns the training design and valid features

Fit SHALL retain the training feature matrix and its missingness mask, and SHALL record
the columns that contain at least one observed value. All-missing columns SHALL be dropped
by default and kept at zero when `keep_empty_features` is enabled.

#### Scenario: All-missing column is dropped

- **WHEN** a column is entirely missing at fit and `keep_empty_features` is false
- **THEN** the transformed output omits that column

#### Scenario: All-missing column is kept at zero

- **WHEN** `keep_empty_features` is true
- **THEN** the transform emits that column with every value zero

### Requirement: Per-column donor selection with NaN-aware distance

For each row that needs imputation and each column with a missing value, the imputer
SHALL consider only training rows with an observed value in that column, compute the
NaN-aware Euclidean distance over co-observed features, and select the
`nearest_neighbors_count` smallest distances.

#### Scenario: Donors must observe the target column

- **WHEN** a training row is missing the column being imputed
- **THEN** that row is not a donor for that column

#### Scenario: Co-observed features drive the distance

- **WHEN** two rows share only a subset of observed features
- **THEN** the distance uses only that subset and scales by the total feature count

#### Scenario: No co-observed feature yields no usable donor

- **WHEN** a receiver has no co-observed feature with any donor
- **THEN** every candidate distance is NaN and the fallback applies

### Requirement: Weighted donor average and fallback

The imputer SHALL fill a missing cell with the uniform or inverse-distance weighted
average of the selected donors' values, and SHALL fall back to the column mean over
observed training values when no donor distance is finite.

#### Scenario: Uniform average

- **WHEN** the three nearest donors hold `2.0`, `4.0` and `6.0`
- **THEN** the imputed value is `4.0`

#### Scenario: Distance-weighted average

- **WHEN** distance weighting is enabled
- **THEN** closer donors contribute more and a zero distance never produces a non-finite
  value

#### Scenario: Column-mean fallback

- **WHEN** every candidate distance is NaN
- **THEN** the cell is filled with the mean of the observed training values of that column

### Requirement: Missingness indicator composition

When `add_indicator` is enabled, the transform SHALL stack the fitted missingness mask
after the imputed features, using the same marker and scope as `SKMissingIndicator`.

#### Scenario: Indicator columns follow the imputed features

- **WHEN** `add_indicator` is true and the input has missing cells
- **THEN** the output has the imputed feature columns followed by one binary column per
  fitted missingness scope feature

#### Scenario: Missingness unseen at fit is tolerated

- **WHEN** a feature complete at fit is missing at transform and `add_indicator` is true
- **THEN** the imputer fills it without failing and the indicator does not add that feature

### Requirement: Bounded-memory and parallel transform

Transform SHALL compute distances in fixed-size chunks so memory stays bounded by the
chunk width, and SHALL process independent receiver rows in parallel while preserving row
order.

#### Scenario: Large input does not allocate a full distance matrix

- **WHEN** a large input is transformed
- **THEN** the computation succeeds without materializing an `n_receivers × n_training`
  matrix

#### Scenario: Parallel equals sequential

- **WHEN** the same input is transformed sequentially and through the parallel path
- **THEN** both outputs are identical

### Requirement: Validation

The imputer SHALL reject empty input, a zero or excessive neighbour count, a
feature-count mismatch and non-finite values other than the configured marker with
structured central errors.

#### Scenario: Feature-count mismatch is rejected

- **WHEN** transform receives a different feature count from fit
- **THEN** it returns `ShapeMismatch` with both dimensions

#### Scenario: Non-finite value is rejected

- **WHEN** a value is infinite or NaN while the marker is a different scalar
- **THEN** the operation returns `NonFiniteValue` naming the position

#### Scenario: Excessive neighbour count is rejected

- **WHEN** `nearest_neighbors_count` exceeds the number of potential donors
- **THEN** the operation returns a structured error instead of silently reducing `k`

### Requirement: Concurrent transform

The fitted model SHALL be safely shareable across threads and SHALL produce the same
output under concurrent transform as under a single thread.

#### Scenario: One model serves several threads

- **WHEN** several threads transform with the same fitted model
- **THEN** every call completes without locking and returns its sequential result
