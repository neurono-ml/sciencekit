# logistic-regression Specification

## Purpose
Defines `SKLogisticRegression`, the linear classifier: a builder-configured, backend-carrying
estimator with a distinct multi-headed fitted model whose objective is the penalized logistic
negative log-likelihood, reachable from dense and sparse inputs, from integer and nominal
labels, and from a dense `fit`, an incremental `partial_fit` and an out-of-core
`fit_streaming`, exposing canonical labels, class probabilities and a decision function.

## Requirements

### Requirement: Builder-only construction with explicit defaults

`SKLogisticRegression<F>` SHALL be constructed only through
`SKLogisticRegression::<F>::build()`, exposing the chainable configuration sink:
`penalty(SKLogisticPenalty)` (default `L2`), `l1_ratio(F)` (default `0.5`), `C(F)` and
`alpha(F)` (default `1.0`), `solver(SKLogisticSolver)` (default `Lbfgs`),
`multi_class(SKMultiClass)` (default `Auto`), `class_weight(SKClassWeight)` (default
`None`), `warm_start(bool)` (default `false`), `include_intercept(bool)` (default `true`),
`tolerance(F)` (default `1e-4`), `maximum_number_of_iterations(usize)` (default `100`),
`backend(SKBackendKind)` (default `Faer`) and `execution_mode(SKExecutionMode)` (default
`Automatic`); the direct constructor SHALL stay private. Because the configuration carries
the scalar `C`/`alpha`, the estimator MAY be generic over `F`.

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it fits with an L2 penalty, regularization strength one, the `Lbfgs` solver,
  automatic multi-class resolution, no class weighting, no warm start, an intercept, carries
  `SKBackendKind::Faer` and an `Automatic` execution intent

#### Scenario: Configuration flows through the model

- **WHEN** an estimator is built with `include_intercept(false)`, a non-default backend and a
  non-default tolerance
- **THEN** the fitted model reports no intercept, carries that backend kind and honors the
  tolerance

### Requirement: Both C and alpha are accepted with last-setter precedence

The builder SHALL expose `C(F)` (inverse regularization) and `alpha(F)` (direct
regularization, `alpha = 1/C`). When only one is set, it SHALL define the effective
regularization strength. When both are set on the same builder, the most recently called
setter SHALL determine the effective value and the build SHALL emit a warning; a
non-positive or non-finite value from either setter SHALL be rejected at build with a
structured error.

#### Scenario: Only C is set

- **WHEN** an estimator is built with `C(0.5)` and no `alpha`
- **THEN** the effective regularization strength is `alpha = 2.0`

#### Scenario: Both are set, alpha last

- **WHEN** an estimator is built with `C(0.5)` then `alpha(4.0)`
- **THEN** the effective regularization strength is `4.0` and a warning is emitted

### Requirement: The objective is the penalized logistic negative log-likelihood

The estimator SHALL minimize the penalized logistic loss: for binary fits
`(1/n)·Σ_i c_i·log(1 + exp(−y_i·x_iᵀw)) + P(w)` with `y_i ∈ {−1, +1}`, and for multinomial
fits the softmax cross-entropy with the same penalty `P(w)`, where `P(w)` is `(α/2)‖w‖₂²`
(`L2`), `α‖w‖₁` (`L1`), `α·(l1_ratio·‖w‖₁ + (1−l1_ratio)/2·‖w‖₂²)` (`ElasticNet`) or `0`
(`None`), with `α = 1/C`. The intercept SHALL never be penalized, except under the
`Liblinear` solver, whose LIBLINEAR formulation regularizes the intercept exactly as
scikit-learn does.

#### Scenario: A large C approaches the unregularized fit

- **WHEN** the same data is fitted with a very large `C` and with `penalty(None)`
- **THEN** the two coefficient vectors agree within tolerance

#### Scenario: Growing the penalty shrinks the coefficients

- **WHEN** the same data is fitted with increasing penalty strength
- **THEN** the coefficient magnitude decreases monotonically

#### Scenario: The intercept is not shrunk

- **WHEN** data whose best separator has a large intercept is fitted with a growing penalty
- **THEN** the intercept stays near its unpenalized estimate while the slopes shrink

### Requirement: Multi-class strategies are resolved deterministically

The estimator SHALL support one-vs-rest and multinomial (softmax) fitting. With
`multi_class(Auto)` it SHALL resolve to multinomial for the smooth solvers (`Lbfgs`, `Saga`)
on more than two classes and to one-vs-rest for `Liblinear` and for binary problems. An
explicit `OvR` or `Multinomial` SHALL force the strategy, and `Liblinear` with `Multinomial`
SHALL return a structured error.

#### Scenario: Auto resolves to multinomial for three classes

- **WHEN** a three-class design is fitted with the `Lbfgs` solver and `multi_class(Auto)`
- **THEN** the model has three coefficient vectors fitted jointly and its probabilities sum
  to one per row

#### Scenario: Liblinear with multinomial is refused

- **WHEN** an estimator pairs `Liblinear` with `multi_class(Multinomial)`
- **THEN** the fit returns a structured error

### Requirement: Targets are canonicalized through the label table

The estimator SHALL accept integer (`i64`) and nominal (`&str`) targets, canonicalize them
through `SKLabelTable`, and reject continuous targets and a single-class target with a
structured error. The fitted model SHALL expose the class table so canonical indices map back
to the original labels in first-occurrence order.

#### Scenario: Nominal labels round-trip

- **WHEN** a design with nominal labels is fitted and predicted
- **THEN** the predicted labels are the original strings and the model reports the class table

#### Scenario: Continuous targets are rejected

- **WHEN** a continuous target vector is passed to `fit`
- **THEN** the fit returns a structured error

#### Scenario: A single class is rejected

- **WHEN** every target belongs to one class
- **THEN** the fit returns a structured error

### Requirement: Prediction returns canonical labels, probabilities and a decision function

The fitted model SHALL implement `SKClassifierPredictor<F>`: `predict_labels` SHALL return
canonical `i64` class indices (the arg-max of the linear predictor or probabilities) and
`predict_probabilities` SHALL return one column per class in the model scalar. The model
SHALL also expose a `decision_function` returning the linear predictor. The estimator SHALL
NOT expose prediction.

#### Scenario: Predicted labels match the classes

- **WHEN** a fitted model predicts a feature row
- **THEN** the label is the class whose probability is largest

#### Scenario: Decision function has one column per class

- **WHEN** a three-class model's decision function is evaluated
- **THEN** it returns one value per class per row

### Requirement: Probabilities are normalized and monotone in the decision function

The predicted probabilities SHALL be the sigmoid of the decision function for binary fits,
the softmax for multinomial fits, and the normalized one-vs-rest sigmoids for one-vs-rest
fits; each row SHALL sum to one within tolerance, lie in `[0, 1]`, and order the classes the
same way as the decision function.

#### Scenario: Rows sum to one

- **WHEN** a fitted model predicts probabilities for any batch
- **THEN** every row sums to one within tolerance and every entry lies in `[0, 1]`

#### Scenario: The decision function orders the classes

- **WHEN** the predicted probabilities and the decision function are compared on the same row
- **THEN** the class with the largest decision value has the largest probability

### Requirement: Class weighting is applied to the loss

The estimator SHALL accept `class_weight(SKClassWeight)` — `None`, `Balanced` or an explicit
per-class weight vector indexed by canonical class order — and apply it as per-sample weights
in the loss, leaving the penalty unweighted. `Balanced` SHALL weight each class by
`n_samples / (n_classes · class_count)`.

#### Scenario: Balanced improves minority recall

- **WHEN** an imbalanced design is fitted with and without the `Balanced` policy
- **THEN** the balanced model predicts the minority class at least as often on the training
  rows

#### Scenario: Explicit weights are applied in class order

- **WHEN** an explicit weight vector is supplied
- **THEN** the fit uses the weight at each sample's canonical class index

### Requirement: Warm start seeds the optimizer from an explicit solution

`warm_start(true)` SHALL seed the `Lbfgs` and `Saga` optimizers from a coefficient solution
supplied on the builder (a previous model's coefficients) instead of the zero vector; the
`Liblinear` path starts cold. The estimator SHALL remain immutable and stateless, so no
solution is reused implicitly across `fit` calls. `warm_start(true)` without a supplied
solution SHALL start from zero.

#### Scenario: Warm start reaches the same optimum

- **WHEN** a model is refitted with `warm_start(true)` seeded from its own solution
- **THEN** the result equals the cold-started fit within tolerance

#### Scenario: Warm start without a source starts from zero

- **WHEN** an estimator is built with `warm_start(true)` and no supplied solution
- **THEN** the fit is identical to the cold-started fit

### Requirement: Sparse input stays sparse

The estimator SHALL accept a zero-copy `CsMatView` design without densifying it, and a sparse
fit's coefficients, intercepts and probabilities SHALL agree with the equivalent dense fit
within the documented relative tolerance.

#### Scenario: Sparse and dense agree

- **WHEN** the same design is fitted densely and as an equivalent CSR matrix
- **THEN** the coefficients, intercepts and probabilities agree within the documented
  relative tolerance

### Requirement: Incremental and streaming fits match the dense fit

`partial_fit` on the fitted model and `fit_streaming` over a lazy source SHALL retain the
design per batch and re-solve, converging to the dense coefficients, intercepts and
probabilities within floating-point tolerance, regardless of batch order or size. An
in-memory execution intent for `fit_streaming` SHALL instead collect the batches and reuse
the dense path. An empty source SHALL be rejected with a structured error.

#### Scenario: Split batches equal the whole

- **WHEN** a dataset is fitted at once and, separately, its batches are accumulated with
  `partial_fit`
- **THEN** the two models' coefficients, intercepts and probabilities agree within tolerance

#### Scenario: Empty source is rejected

- **WHEN** a streaming source emits no batches
- **THEN** the fit returns a structured error and no model

### Requirement: Structured errors on invalid input

The estimator SHALL reject an empty design, a target count disagreeing with the row count, a
non-positive regularization strength, a non-positive `tolerance`, a zero iteration budget, an
unsupported solver/penalty combination, an out-of-range `l1_ratio`, a single-class target and
continuous targets, each with the central `SKError` taxonomy, and prediction SHALL reject a
column count differing from the fitted design.

#### Scenario: Empty input

- **WHEN** `fit` receives zero rows
- **THEN** it returns a structured error before touching the solver

#### Scenario: Prediction shape mismatch

- **WHEN** the model predicts on a matrix whose column count differs from the fitted design
- **THEN** it returns a shape-mismatch error

### Requirement: Backend and execution intent are carried and recorded

Per the `backend-selection` capability, every estimator and model operation SHALL record the
configured backend kind in its observability span; `fit` SHALL transfer the backend kind into
the fitted model, and the kind SHALL never be hardcoded.

#### Scenario: Configured kind reaches the model

- **WHEN** an estimator is built with a non-default backend and fitted
- **THEN** the model reports that backend kind and its span records the kind

### Requirement: The fitted model is shareable across threads

The fitted model SHALL be `Send + Sync` by construction so a single model serves concurrent
predictions without external synchronization.

#### Scenario: One model, many threads

- **WHEN** the same fitted model predicts on distinct inputs from multiple threads
- **THEN** every prediction matches the single-threaded result
