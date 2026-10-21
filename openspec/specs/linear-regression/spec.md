# linear-regression Specification

## Purpose
Defines `SKLinearRegression`, the first supervised estimator: a builder-configured,
backend-carrying ordinary-least-squares regressor with a distinct fitted model, reachable
identically from a dense `fit`, an incremental `partial_fit` and an out-of-core
`fit_streaming`.

## Requirements

### Requirement: Builder-only construction with explicit defaults

`SKLinearRegression` SHALL be constructed only through `SKLinearRegression::build()`, which
SHALL expose chainable `include_intercept(bool)` (defaulting to `true`),
`backend(SKBackendKind)` (defaulting to `SKBackendKind::Faer`) and
`execution_mode(SKExecutionMode)` (defaulting to `Automatic`); the direct constructor SHALL
stay private. The configuration carries no scalar value, so the estimator is not generic
over `F`.

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it fits an intercept, carries `SKBackendKind::Faer` and an `Automatic` execution
  intent

#### Scenario: Configuration override flows through the model

- **WHEN** an estimator is built with `include_intercept(false)` and a non-default backend
- **THEN** the fitted model reports no intercept and carries that backend kind

### Requirement: Dense fit learns coefficients by least squares

`fit` SHALL learn the coefficient vector that minimizes the squared error between the
fitted responses and the targets, returning a fitted model that exposes its coefficients,
intercept, rank and singular values. The solve SHALL go through the backend's least-squares
kernel (the singular-value / pseudoinverse path), never through the normal equations.

#### Scenario: Exact fit on a hand-checkable line

- **WHEN** `y = 2x + 1` is fitted with the default configuration
- **THEN** the model reports coefficient `2` and intercept `1` within tolerance

#### Scenario: Rank and singular values are surfaced

- **WHEN** a full-rank design is fitted
- **THEN** the model reports a rank equal to the design width and a non-increasing list of
  singular values

### Requirement: Intercept by design-matrix augmentation

When `include_intercept` is true, the fit SHALL prepend a column of ones to the design
matrix so the intercept is solved jointly with the coefficients; when false, the model's
intercept SHALL be zero and no column SHALL be added.

#### Scenario: Intercept absorbed into the joint solve

- **WHEN** data whose best line has a non-zero intercept is fitted with the default
  configuration
- **THEN** the reported intercept matches the true one within tolerance

#### Scenario: Intercept disabled

- **WHEN** the same data is fitted with `include_intercept(false)`
- **THEN** the model reports a zero intercept and fits the best line through the origin

### Requirement: Rank-deficient designs yield the minimum-norm solution

When the design matrix is rank-deficient, `fit` SHALL still succeed and return the
minimum-norm least-squares solution, surfacing the effective rank below the design width.

#### Scenario: Collinear columns

- **WHEN** a design with a duplicated column is fitted
- **THEN** the fit succeeds, the reported rank is below the design width, and the residuals
  equal those of the equivalent full-rank model

### Requirement: Prediction returns responses in the model scalar

The fitted model SHALL implement regressor prediction, computing each response as the dot
product of the feature row with the coefficients plus the intercept, and returning an owned
`Array1<F>` in the model scalar. The estimator SHALL NOT expose prediction.

#### Scenario: Predictions on new rows

- **WHEN** the model fitted on `y = 2x + 1` predicts a new feature row
- **THEN** the response equals `2·x + 1` within tolerance

#### Scenario: Both supported floats

- **WHEN** the same data is fitted and predicted as `f32` and as `f64`
- **THEN** both succeed and return responses in their own scalar

### Requirement: Incremental fit matches the concatenated fit

`partial_fit` on the fitted model SHALL merge one more batch into running cross-product
accumulators and converge to the same coefficients as a single fit on the concatenated
data, regardless of batch order or size, within floating-point tolerance.

#### Scenario: Split batches equal the whole

- **WHEN** a dataset is fitted at once and, separately, its batches are accumulated with
  `partial_fit`
- **THEN** the two models' coefficients and intercept agree within tolerance

#### Scenario: Empty batch is a no-op

- **WHEN** an empty batch is passed to `partial_fit`
- **THEN** the model is unchanged

### Requirement: Streaming fit honors the execution intent

`fit_streaming` SHALL consume a lazy source through the streaming driver, accumulating the
same cross-product state per batch and solving once at the end, so the streaming model
equals the dense fit within tolerance. An in-memory execution intent SHALL instead collect
the batches and reuse the dense path.

#### Scenario: Uneven batches match the dense fit

- **WHEN** a source emits uneven batches of the same data fitted densely
- **THEN** the streaming model's coefficients and intercept agree with the dense fit within
  tolerance

#### Scenario: Empty source is rejected

- **WHEN** a source emits no batches
- **THEN** the fit returns a structured error and no model

### Requirement: Structured errors on invalid input

The estimator SHALL reject empty inputs and sparse inputs before processing any element,
and prediction SHALL reject a column count differing from the fitted design, each with the
central `SKError` taxonomy. A target count differing from the feature row count SHALL also
be rejected.

#### Scenario: Empty input

- **WHEN** `fit` receives zero rows
- **THEN** it returns a structured error before touching the backend

#### Scenario: Sparse input is unsupported here

- **WHEN** `fit` receives a sparse matrix
- **THEN** it returns an `UnsupportedRepresentation` error

#### Scenario: Prediction shape mismatch

- **WHEN** the model predicts on a matrix whose column count differs from the fitted design
- **THEN** it returns a `ShapeMismatch` error

### Requirement: Backend and execution intent are carried and recorded

Per the `backend-selection` capability, every estimator and model operation SHALL record
the configured backend kind in its observability span, and `fit` SHALL transfer the kind
into the fitted model; the kind SHALL never be hardcoded.

#### Scenario: Configured kind reaches the model

- **WHEN** an estimator is built with a non-default backend and fitted
- **THEN** the model reports that backend kind

### Requirement: The fitted model is shareable across threads

The fitted model SHALL be `Send + Sync` by construction so a single model serves concurrent
predictions without external synchronization.

#### Scenario: One model, many threads

- **WHEN** the same fitted model predicts on distinct inputs from multiple threads
- **THEN** every prediction matches the single-threaded result
