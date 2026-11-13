# Spec Delta

## Purpose

Defines the linear support-vector classifier: its builder surface, the dual
coordinate-descent fitting contract for the two supported losses, one-vs-rest
multiclass behavior, class weights, decision and label outputs, representation parity and
convergence semantics.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

`SKLinearSVC` SHALL be constructible only through its builder, exposing `C` (default
1.0), `loss` (`SquaredHinge` default, `Hinge`), `class_weight` (`None` default),
`include_intercept` (default true), `tolerance`, `maximum_number_of_iterations`, `backend`
(default `Faer`) and `execution_mode` (default `Automatic`).

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it uses `C = 1.0`, squared hinge, no class weights, an intercept, `Faer` and
  `Automatic`

#### Scenario: Backend reaches the fitted model

- **WHEN** a non-default backend is configured and the estimator is fitted
- **THEN** the fitted model records that backend in its prediction spans

### Requirement: Dual coordinate-descent fitting

The classifier SHALL fit the L2-regularized linear SVC dual by coordinate descent, with
the configured loss, `C`, tolerance and iteration budget, and SHALL return
`NotConverged { iterations }` when the budget is exhausted.

#### Scenario: Separable binary data is separated

- **WHEN** two linearly separable classes are fitted
- **THEN** every training row is classified correctly

#### Scenario: Exhausted budget is reported

- **WHEN** the iteration budget is too small to reach the tolerance
- **THEN** fit returns `NotConverged` carrying the executed iteration count

### Requirement: Loss selection

The classifier SHALL support the L1-hinge loss and the L2-squared-hinge loss, and the
chosen loss SHALL change the fitted objective without changing the builder surface.

#### Scenario: Squared hinge is the default

- **WHEN** an estimator is built without setting `loss`
- **THEN** the model reports the squared-hinge loss

#### Scenario: Hinge loss is honored

- **WHEN** `loss(Hinge)` is configured
- **THEN** the model reports the hinge loss and fits the corresponding objective

### Requirement: One-vs-rest multiclass classification

For more than two classes the classifier SHALL fit one binary model per class and SHALL
predict the class with the largest decision value, breaking ties by the smallest
canonical class index.

#### Scenario: Three classes produce three heads

- **WHEN** a three-class problem is fitted
- **THEN** the model retains three coefficient vectors and three intercepts

#### Scenario: Decision ties are deterministic

- **WHEN** two classes share the largest decision value
- **THEN** the class with the smaller canonical index is predicted

### Requirement: Class weights

The classifier SHALL support `None`, `Balanced` and explicit per-class weights, and the
weights SHALL change the effective regularization of the affected class.

#### Scenario: Balanced weights on an imbalanced fixture

- **WHEN** `Balanced` is configured on an imbalanced data set
- **THEN** the fitted boundary differs from the unweighted fit

#### Scenario: Explicit weights are honored

- **WHEN** an explicit weight map is configured
- **THEN** every class uses its configured weight

### Requirement: Decision function and labels

The model SHALL expose a decision function with one value per class and SHALL predict
labels from it through the canonical class table.

#### Scenario: Decision function shape

- **WHEN** a batch of `m` rows is scored on a `c`-class model
- **THEN** the decision matrix has shape `(m, c)`

#### Scenario: Labels match the argmax

- **WHEN** labels and decisions are requested for the same batch
- **THEN** each predicted label corresponds to the largest decision value

### Requirement: Probabilities are unsupported

`predict_probabilities` SHALL return `UnsupportedOperation` with guidance to a calibrated
classifier, because the linear SVC has no probability model.

#### Scenario: Probability request is rejected

- **WHEN** probabilities are requested from a fitted `SKLinearSVC`
- **THEN** the call returns `UnsupportedOperation` and never fabricates a distribution

### Requirement: Representation parity

`SKLinearSVC` SHALL accept dense or borrowed CSR features for fit and prediction, dispatch
once per operation and never densify a sparse design.

#### Scenario: Sparse fit equals dense fit

- **WHEN** the same logical design is fitted densely and as CSR
- **THEN** the coefficients and predictions agree within the documented tolerance

#### Scenario: Sparse prediction is supported

- **WHEN** a model predicts on a sparse batch
- **THEN** prediction reads the stored entries and succeeds without densifying

### Requirement: Determinism and concurrency

A fixed configuration and input SHALL produce identical models and predictions across
repeated fits and thread counts, and a fitted model SHALL be safely shareable.

#### Scenario: Repeated fits agree

- **WHEN** the same estimator fits the same data twice
- **THEN** coefficients, intercepts and predictions are identical

#### Scenario: Concurrent prediction agrees

- **WHEN** several threads predict with the same fitted model
- **THEN** every result equals the sequential result

### Requirement: Validation

The classifier SHALL reject empty input, single-class targets, feature-count mismatch,
non-finite values, non-positive `C`, zero tolerance or iteration budget, and an
unsupported loss before fitting.

#### Scenario: Single-class target is rejected

- **WHEN** fit receives only one class
- **THEN** it returns `InvalidHyperparameter` naming the class requirement

#### Scenario: Non-positive C is rejected

- **WHEN** `C` is zero or negative
- **THEN** construction returns `InvalidHyperparameter` naming `C`

#### Scenario: Feature-count mismatch is rejected

- **WHEN** prediction receives a different feature count from fit
- **THEN** it returns `ShapeMismatch` with both dimensions
