# Spec Delta

## Purpose

Defines `SKSVC` and `SKSVR`: the builder surface, the SMO fitting contract, multiclass
one-vs-one behavior, probability calibration, support-vector model state, representation
parity and convergence semantics shared by the two estimators.

## ADDED Requirements

### Requirement: Builder-only estimator configuration

`SKSVC` and `SKSVR` SHALL be constructed only through their builders, and every builder
SHALL expose `backend` defaulting to `Faer` and `execution_mode` defaulting to
`Automatic`, plus the estimator-specific hyperparameters.

#### Scenario: Defaults are applied without configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it carries `SKBackendKind::Faer` and `SKExecutionMode::Automatic`

#### Scenario: Backend override reaches the fitted model

- **WHEN** an estimator is built with a non-default backend and fitted
- **THEN** the fitted model records that backend in its prediction spans

### Requirement: C-SVC fitting

`SKSVC` SHALL fit a C-support-vector classifier with the configured kernel, regularization
strength, tolerance, iteration budget and class weights, and SHALL retain the support
vectors and dual coefficients needed for prediction.

#### Scenario: Linearly separable data is separated

- **WHEN** two separable classes are fitted with a linear kernel
- **THEN** every training row is classified correctly

#### Scenario: Support vectors are a subset of training rows

- **WHEN** a fit completes
- **THEN** every retained support vector equals a training row and the dual coefficients
  have the same length

### Requirement: One-vs-one multiclass classification

`SKSVC` SHALL train one binary classifier per pair of classes and SHALL resolve a
prediction by majority vote, breaking ties by the smallest canonical class index.

#### Scenario: Three classes produce three pairwise classifiers

- **WHEN** a three-class problem is fitted
- **THEN** exactly three binary decision functions are retained

#### Scenario: Vote ties are deterministic

- **WHEN** two classes receive the same number of votes
- **THEN** the class with the smaller canonical index is predicted

### Requirement: Class probability calibration

`SKSVC` SHALL expose class probabilities calibrated from the binary decision values, and
SHALL return one probability per class for every input row.

#### Scenario: Probabilities sum to one

- **WHEN** probabilities are requested for a batch
- **THEN** each row sums to `1` within the documented tolerance and every value is in
  `[0, 1]`

#### Scenario: Binary calibration uses Platt scaling

- **WHEN** a binary problem is fitted with probability estimation enabled
- **THEN** the probability increases monotonically with the decision value

#### Scenario: Multiclass probabilities use pairwise coupling

- **WHEN** a multiclass problem is fitted with probability estimation enabled
- **THEN** the returned probabilities are the coupled pairwise estimates, not normalized
  vote counts

### Requirement: Epsilon-support-vector regression

`SKSVR` SHALL fit an epsilon-insensitive support-vector regressor with the configured
kernel, regularization strength, epsilon band, tolerance and iteration budget.

#### Scenario: Predictions follow the fitted response surface

- **WHEN** a regressor is fitted to a smooth one-dimensional response
- **THEN** predictions on the training rows are within the documented tolerance of the
  targets

#### Scenario: Epsilon band controls support vectors

- **WHEN** the epsilon band is widened on the same data
- **THEN** the number of retained support vectors does not increase

### Requirement: Kernel selection

`SKSVC` and `SKSVR` SHALL accept the linear, polynomial, radial-basis-function and sigmoid
kernels with their documented parameters, and SHALL reject an unknown kernel or invalid
parameter before fitting.

#### Scenario: Kernel choice changes the decision boundary

- **WHEN** the same non-linear data is fitted with a linear and an RBF kernel
- **THEN** the RBF model fits at least as well on the training rows

#### Scenario: Invalid kernel parameter is rejected

- **WHEN** a negative `gamma` or a non-positive degree is configured
- **THEN** construction returns `InvalidHyperparameter` naming the parameter

### Requirement: Representation parity

`SKSVC` and `SKSVR` SHALL accept dense, sparse and mixed dense/sparse feature inputs for
fit and prediction, SHALL dispatch once per operation and SHALL never densify a sparse
input.

#### Scenario: Sparse fit equals dense fit

- **WHEN** the same logical training matrix is fitted once densely and once as CSR
- **THEN** the fitted models produce the same predictions within the documented tolerance

#### Scenario: Mixed prediction is supported

- **WHEN** a model fitted on sparse data predicts on a dense batch, or the reverse
- **THEN** prediction succeeds without densifying the sparse side

### Requirement: Convergence and iteration budget

The SMO solver SHALL stop when the maximum KKT violation falls below the configured
tolerance, and SHALL return `NotConverged` with the executed iteration count when the
budget is exhausted.

#### Scenario: Converged fit reports convergence

- **WHEN** the solver reaches the tolerance
- **THEN** the fitted model reports convergence and the iterations executed

#### Scenario: Exhausted budget is reported

- **WHEN** the iteration budget is too small to reach the tolerance
- **THEN** fitting returns `NotConverged` carrying the number of iterations executed

### Requirement: Deterministic fitting

For a fixed configuration and a fixed input, `SKSVC` and `SKSVR` SHALL produce identical
models and predictions across repeated fits and across thread counts.

#### Scenario: Repeated fits agree

- **WHEN** the same estimator fits the same data twice
- **THEN** the support vectors, dual coefficients and predictions are identical

#### Scenario: Parallel prediction agrees with sequential prediction

- **WHEN** a large batch is predicted through the row-parallel path and sequentially
- **THEN** both results are identical

### Requirement: Input validation

`SKSVC` and `SKSVR` SHALL reject empty inputs, feature-count mismatches, non-finite
values and fewer than two distinct classes for classification, before starting the solver.

#### Scenario: Empty input is rejected

- **WHEN** fit receives zero rows
- **THEN** it returns `EmptyInput` naming the operation

#### Scenario: Single-class classification is rejected

- **WHEN** `SKSVC` receives only one class
- **THEN** fit returns `InvalidHyperparameter` naming the class requirement

#### Scenario: Non-finite value is rejected

- **WHEN** fit or prediction receives NaN or infinity
- **THEN** the operation returns `NonFiniteValue` naming the position
