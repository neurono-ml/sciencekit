# Spec Delta

## Purpose

Defines `SKRobustScaler`, the transformer that centers and scales features on quantile
statistics (median and interquantile range) so outliers do not dominate the scale, with
incremental and out-of-core fitting through an online quantile sketch.

## ADDED Requirements

### Requirement: Builder-only construction with explicit defaults

`SKRobustScaler` SHALL be constructed only through `SKRobustScaler::build()`, exposing
chainable `with_centering(bool)` and `with_scaling(bool)` (both defaulting to `true`),
`quantile_range((F, F))` defaulting to `(25.0, 75.0)`, `unit_variance(bool)` defaulting to
`false`, `backend(SKBackendKind)` defaulting to `Faer` and `execution_mode(SKExecutionMode)`
defaulting to `Automatic`; the direct constructor SHALL stay private. The estimator SHALL stay
generic over the scalar `F: SKFloat`.

#### Scenario: Default configuration

- **WHEN** a scaler is built without calling any setter
- **THEN** it centers on the median, scales by the 25–75 interquartile range, does not
  unit-variance adjust, and carries `Faer` and an `Automatic` intent

### Requirement: Dense fit uses exact per-column quantiles

`fit` SHALL learn, for each column, the median (the center) and the interquantile range over
the configured `quantile_range` (the scale), computed exactly from the dense data (matching
scikit-learn), and the fitted model SHALL expose them.

#### Scenario: Median and interquartile range

- **WHEN** a column with values `[1, 2, 3, 4, 5]` is fitted with the default range `(25, 75)`
- **THEN** the center is `3` and the scale is `2` (the 75th minus the 25th percentile)

### Requirement: Transform centers and scales with masking

`transform` SHALL map each value as $z = (x - \text{center}) / \text{scale}$, omitting the
centering when `with_centering` is false and the scaling when `with_scaling` is false; a
non-positive scale SHALL be treated as one so the output stays finite.

#### Scenario: Centering and scaling

- **WHEN** a model is fitted with both flags true and transforms the training data
- **THEN** the median maps to `0` and the interquartile range maps to `1`

#### Scenario: Masking disables an axis

- **WHEN** `with_centering` is false, transform SHALL NOT center; when `with_scaling` is
  false, it SHALL NOT scale
- **THEN** the corresponding operation is the identity on that axis

#### Scenario: Zero-spread column

- **WHEN** a column has zero interquantile range
- **THEN** its scale is one and its transformed values are finite

### Requirement: Unit-variance adjustment for normally distributed data

When `unit_variance` is true, the scale SHALL be divided by the standard-normal interquantile
span of the same `quantile_range`, so that normally distributed features are scaled to unit
variance.

#### Scenario: Standard-normal span

- **WHEN** `unit_variance` is true and `quantile_range` is `(25, 75)`
- **THEN** the scale is divided by the standard-normal 75th-minus-25th percentile span
  (approximately `1.34898`)

### Requirement: Structured errors on invalid input

`fit` SHALL reject empty inputs and sparse inputs before processing any element, and
`transform` SHALL reject a column count differing from the fitted model, each with the central
`SKError` taxonomy.

#### Scenario: Empty input

- **WHEN** `fit` receives a matrix with zero rows or zero columns
- **THEN** it returns a `ShapeMismatch` error before computing anything

#### Scenario: Sparse input

- **WHEN** `fit` receives a sparse matrix
- **THEN** it returns an `UnsupportedRepresentation` error

#### Scenario: Transform column mismatch

- **WHEN** a fitted model transforms data whose column count differs from its own
- **THEN** it returns a `ShapeMismatch` error and produces no output

### Requirement: Incremental partial_fit uses an online quantile sketch

The fitted model SHALL accept `partial_fit` over additional batches, maintaining an online
quantile sketch per column, so that the center and scale converge to the dense result within a
documented approximation tolerance; `sample_count` SHALL accumulate. An empty batch SHALL be a
no-op; a column-count mismatch SHALL fail without touching the model.

#### Scenario: Incremental approximates dense

- **WHEN** a model is fitted on part of the data and `partial_fit` on the remainder
- **THEN** its center and scale approximate the dense fit's within the documented tolerance

#### Scenario: Empty batch is a no-op

- **WHEN** `partial_fit` receives a batch with zero rows
- **THEN** neither the statistics nor the sample count change

#### Scenario: Mismatched batch is rejected

- **WHEN** `partial_fit` receives a batch with the wrong column count
- **THEN** it returns a `ShapeMismatch` error and the model is unchanged

### Requirement: Out-of-core fit_streaming approximates the dense fit

`fit_streaming` SHALL consume a sequential `SKLazySource` and produce a model whose center and
scale approximate the dense fit's within the documented tolerance; under an in-memory
execution intent it SHALL collect the batches and reuse the dense path, otherwise it SHALL
feed each batch into the online sketch through the streaming executor. An empty source or
inconsistent column counts SHALL fail with a structured error.

#### Scenario: Uneven batches approximate dense

- **WHEN** the data is streamed in uneven batches and compared with the dense fit
- **THEN** the center and scale approximate the dense fit's within tolerance

### Requirement: The configured backend is carried and recorded

Every operation SHALL record the configured `SKBackendKind` in its observability span — never
a hardcoded value — and `fit` SHALL transfer the kind into the fitted model, whose operations
record the carried kind.

#### Scenario: Model inherits the estimator backend

- **WHEN** an estimator built with a non-default backend is fitted
- **THEN** the fitted model reports and records that same backend kind
