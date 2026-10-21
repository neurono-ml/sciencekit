# minmax-scaler Specification

## Purpose
Defines `SKMinMaxScaler`, the transformer that maps every feature linearly onto a
configurable target range, with incremental and out-of-core fitting.

## Requirements

### Requirement: Builder-only construction with explicit defaults

`SKMinMaxScaler` SHALL be constructed only through `SKMinMaxScaler::build()`, exposing
chainable `feature_range((F, F))` defaulting to `(0, 1)`, `clip(bool)` defaulting to `false`,
`backend(SKBackendKind)` defaulting to `Faer` and `execution_mode(SKExecutionMode)` defaulting
to `Automatic`; the direct constructor SHALL stay private. The estimator SHALL stay generic
over the scalar `F: SKFloat`.

#### Scenario: Default configuration

- **WHEN** a scaler is built without calling any setter
- **THEN** it maps features to `[0, 1]`, does not clip, carries `Faer` and an `Automatic` intent

#### Scenario: Feature-range override

- **WHEN** a scaler is built with `feature_range((-1, 1))`
- **THEN** transformed features lie in `[-1, 1]` for data within the fitted range

### Requirement: Dense fit learns per-column minimum and range

`fit` SHALL learn, for each column, the minimum and the maximum (the "range" being
`maximum - minimum`), returning a fitted model that exposes the learned minimum, range and the
derived affine parameters. Transformation SHALL be affine: for a feature value $x$,

$$x_{\text{scaled}} = (x - x_{\min})\,\frac{r_{\max} - r_{\min}}{x_{\max} - x_{\min}} + r_{\min},$$

where $(r_{\min}, r_{\max})$ is the configured `feature_range`.

#### Scenario: Mapping to the unit range

- **WHEN** a column with values `[0, 10]` is fitted with the default range `[0, 1]`
- **THEN** the minimum maps to `0`, the maximum maps to `1`, and a midpoint maps to `0.5`

### Requirement: Constant features stay finite

A column whose minimum equals its maximum SHALL be treated as having unit scale and a
target-range minimum, so its transformed values are the target minimum rather than `NaN`.

#### Scenario: Constant column

- **WHEN** a matrix with a constant column is fitted and transformed
- **THEN** that column's transformed values all equal the target-range minimum

### Requirement: Clip constrains output to the feature range

When `clip` is true, transformed values SHALL be clipped to the configured `feature_range`;
when false, values outside the fitted range SHALL be allowed to fall outside the target range.

#### Scenario: Out-of-range value with clipping

- **WHEN** `clip` is true and a value larger than the fitted maximum is transformed
- **THEN** the output equals the target-range maximum

#### Scenario: Out-of-range value without clipping

- **WHEN** `clip` is false and the same value is transformed
- **THEN** the output exceeds the target-range maximum

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

### Requirement: Incremental partial_fit converges to the dense result

The fitted model SHALL accept `partial_fit` over additional batches, updating the running
per-column minima and maxima, so that fitting a prefix and then `partial_fit`-ing the rest
equals the dense fit over the whole data (min/max are exact and order-independent). An empty
batch SHALL be a no-op; a column-count mismatch SHALL fail without touching the model.

#### Scenario: Prefix plus partial_fit equals dense

- **WHEN** a model is fitted on part of the data and `partial_fit` on the remainder
- **THEN** its learned minima, ranges and transform output equal the dense fit's exactly

#### Scenario: Empty batch is a no-op

- **WHEN** `partial_fit` receives a batch with zero rows
- **THEN** neither the statistics nor the sample count change

### Requirement: Out-of-core fit_streaming equals the dense fit

`fit_streaming` SHALL consume a sequential `SKLazySource` and produce the same model the dense
`fit` would produce for the concatenated data; under an in-memory execution intent it SHALL
collect the batches and reuse the dense path, otherwise it SHALL update running minima/maxima
per batch through the streaming executor. An empty source or inconsistent column counts SHALL
fail with a structured error.

#### Scenario: Uneven batches equal dense

- **WHEN** the data is streamed in uneven batches and compared with the dense fit
- **THEN** the learned parameters and transform output equal the dense fit's exactly

### Requirement: The configured backend is carried and recorded

Every operation SHALL record the configured `SKBackendKind` in its observability span — never
a hardcoded value — and `fit` SHALL transfer the kind into the fitted model, whose operations
record the carried kind.

#### Scenario: Model inherits the estimator backend

- **WHEN** an estimator built with a non-default backend is fitted
- **THEN** the fitted model reports and records that same backend kind
