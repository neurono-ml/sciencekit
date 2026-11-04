## Purpose

Defines `SKSGDRegressor`, the stochastic-gradient linear regressor: a builder-configured,
backend-carrying estimator with a distinct fitted model that learns continuous responses from
dense, sparse and out-of-core streams one sample at a time, including robust and
insensitive-band losses.

## ADDED Requirements

### Requirement: Builder-only construction with explicit defaults

`SKSGDRegressor<F>` SHALL be constructed only through `SKSGDRegressor::<F>::build()`; the
direct constructor SHALL stay private. The builder SHALL expose a chainable configuration
sink covering the loss, penalty, regularization, schedule, convergence, stopping, shuffling,
averaging, warm start, backend and execution knobs, plus the `epsilon` band.

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it fits with the squared-error loss, an L2 penalty, `alpha` one ten-thousandth,
  an intercept, the inverse-scaling schedule starting at one hundredth, an `epsilon` of
  one tenth, no early stopping, the `Strict` convergence policy, shuffling with seed zero,
  no averaging, `SKBackendKind::Faer` and an `Automatic` execution intent

#### Scenario: Every knob is settable

- **WHEN** each setter is called with a non-default value
- **THEN** the built estimator exposes that value and the fitted model reflects it

#### Scenario: Configuration flows through the model

- **WHEN** an estimator is built with `include_intercept(false)`, a non-default loss and a
  non-default backend
- **THEN** the fitted model reports no intercept, that loss and that backend kind

### Requirement: Targets are continuous responses

The estimator SHALL accept continuous and integer targets, promoting integers to the model
scalar, and SHALL reject nominal targets with a structured error. A target length differing
from the number of rows SHALL be rejected with a shape error.

#### Scenario: Integer responses are promoted

- **WHEN** an integer target vector is fitted
- **THEN** the fit succeeds and predictions are continuous

#### Scenario: Nominal targets are rejected

- **WHEN** a nominal target vector is passed to `fit`
- **THEN** the fit returns a structured error

### Requirement: Each regressor loss has its documented behavior

The estimator SHALL support `squared_error`, `huber`, `epsilon_insensitive` and
`squared_epsilon_insensitive`. `huber` SHALL be quadratic inside the `epsilon` band and
linear beyond it; the insensitive losses SHALL ignore residuals inside the `epsilon` band.
`epsilon` SHALL be consulted only by these three losses. A very large `epsilon` with `huber`
SHALL approach half the squared error, and a zero `epsilon` with `epsilon_insensitive` SHALL
equal the absolute error.

#### Scenario: Large epsilon reproduces squared error

- **WHEN** `huber` is fitted with an `epsilon` far larger than every residual
- **THEN** the coefficients match a `squared_error` fit within tolerance

#### Scenario: The insensitive band ignores small residuals

- **WHEN** `epsilon_insensitive` is fitted with an `epsilon` larger than every residual
- **THEN** the coefficients stay at their starting values

#### Scenario: Huber resists outliers

- **WHEN** a design is corrupted with an extreme response and fitted with `huber` and with
  `squared_error`
- **THEN** the `huber` coefficients move less from the clean fit

### Requirement: Prediction returns continuous responses

The fitted model SHALL implement the regressor prediction contract `predict` returning one
response per row, computing `X·coefficients + intercept`, and SHALL preserve the model scalar
end to end.

#### Scenario: Prediction matches the linear predictor

- **WHEN** predictions are requested for a known design and coefficients
- **THEN** each response equals the dot product plus the intercept within tolerance

#### Scenario: Prediction shape mismatch is rejected

- **WHEN** prediction receives a design with a different feature count
- **THEN** it returns a shape error

### Requirement: Dense, sparse, streaming and incremental learning share one model

`fit` SHALL accept dense and CSR designs; `fit_streaming` SHALL consume a supervised one-shot
source in a single pass without retaining the design; `partial_fit` on the fitted model SHALL
continue the schedule and the average. The streaming and incremental fits SHALL agree with
the equivalent in-memory epochs within the documented tolerance under the same seed.

#### Scenario: One streaming pass equals one in-memory epoch

- **WHEN** the same data is fitted in one in-memory epoch and streamed once in the same order
  without shuffling
- **THEN** the coefficients agree within tolerance

#### Scenario: Sparse equals dense under a fixed seed

- **WHEN** the same design is fitted densely and as an equivalent CSR matrix with the same
  seed and shuffling disabled
- **THEN** the coefficients agree within tolerance

#### Scenario: Incremental continuation updates the response surface

- **WHEN** `partial_fit` is called with new data after `fit`
- **THEN** the fit on the concatenated data and the incremental fit agree within tolerance

### Requirement: The estimator approaches least squares without regularization

With `squared_error`, a zero or negligible penalty and enough epochs, the fitted coefficients
SHALL approach the ordinary least-squares solution obtained by `SKLinearRegression` within a
documented tolerance.

#### Scenario: SGD converges to the OLS solution

- **WHEN** a well-conditioned design is fitted with `squared_error`, `alpha(0.0)` and a
  generous epoch budget
- **THEN** the coefficients agree with `SKLinearRegression` within tolerance

### Requirement: The fitted model reports its convergence state

The fitted model SHALL expose whether the fit met the convergence criterion and how many
epochs it executed. A permissive fit that stopped short SHALL be distinguishable from a
converged one without inspecting logs.

#### Scenario: A converged model says so

- **WHEN** a fit meets the plateau criterion
- **THEN** the model reports convergence and the epochs it took

#### Scenario: A pass-limited model says it did not converge

- **WHEN** a streaming fit is limited to fewer passes than its budget
- **THEN** the model reports that it did not converge and the passes it took

### Requirement: Every operation records the configured backend

Every public operation SHALL run inside an observability span recording the configured
`SKBackendKind`, and `fit` SHALL transfer the kind into the fitted model, whose own
operations record it. The estimator and the model SHALL be immutable after construction and
shareable across threads.

#### Scenario: Fit transfers the backend kind

- **WHEN** an estimator built with a non-default backend is fitted
- **THEN** the model reports that backend kind and its predictions record it

#### Scenario: One model, many threads

- **WHEN** one fitted model serves concurrent predictions
- **THEN** every call returns the same result without interior mutation
