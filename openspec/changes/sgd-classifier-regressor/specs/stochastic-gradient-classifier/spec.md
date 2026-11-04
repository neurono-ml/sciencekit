## Purpose

Defines `SKSGDClassifier`, the stochastic-gradient linear classifier: a builder-configured,
backend-carrying estimator with a distinct one-vs-rest fitted model that learns from dense,
sparse and out-of-core streams one sample at a time, exposing canonical labels, a decision
function and probabilities for the probabilistic losses.

## ADDED Requirements

### Requirement: Builder-only construction with explicit defaults

`SKSGDClassifier<F>` SHALL be constructed only through `SKSGDClassifier::<F>::build()`; the
direct constructor SHALL stay private. The builder SHALL expose a chainable configuration
sink covering the loss, penalty, regularization, schedule, convergence, stopping, shuffling,
averaging, weighting, warm start, backend and execution knobs.

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it fits with the hinge loss, an L2 penalty, `alpha` one ten-thousandth, an
  intercept, the optimal schedule, no early stopping, the `Strict` convergence policy,
  shuffling with seed zero, no averaging, no class weighting, `SKBackendKind::Faer` and an
  `Automatic` execution intent

#### Scenario: Every knob is settable

- **WHEN** each setter is called with a non-default value
- **THEN** the built estimator exposes that value and the fitted model reflects it

#### Scenario: Configuration flows through the model

- **WHEN** an estimator is built with `include_intercept(false)`, averaging enabled and a
  non-default backend
- **THEN** the fitted model reports no intercept, averaged coefficients and that backend kind

### Requirement: Multi-class fits are one-vs-rest

The estimator SHALL fit one binary head per class and SHALL predict the class with the
largest decision value. A binary problem SHALL produce a single head. No multinomial/softmax
strategy SHALL be offered.

#### Scenario: Three classes produce three heads

- **WHEN** a three-class design is fitted
- **THEN** the model has three coefficient vectors and `predict_labels` returns the arg-max
  head

#### Scenario: Binary fits one head

- **WHEN** a two-class design is fitted
- **THEN** the model has one coefficient vector and predicts both classes

### Requirement: Targets are canonicalized through the label table

The estimator SHALL accept integer (`i64`) and nominal (`&str`) targets and canonicalize
them through `SKLabelTable`; continuous targets and a single-class target SHALL be rejected
with a structured error. The fitted model SHALL expose the class table, and the canonical
order SHALL be first occurrence.

#### Scenario: Nominal labels round-trip

- **WHEN** a design with nominal labels is fitted and predicted
- **THEN** the predicted labels are the original strings and the model reports the class table

#### Scenario: Continuous targets are rejected

- **WHEN** a continuous target vector is passed to `fit`
- **THEN** the fit returns a structured error

### Requirement: Prediction exposes labels, decisions and probabilistic output

The fitted model SHALL implement the classifier prediction contract: `predict_labels`
returning canonical indices and `predict_probabilities` returning one row per sample.
`predict_probabilities` SHALL be available only for `log_loss` and `modified_huber`, whose
rows SHALL sum to one; for any other loss it SHALL return
`SKError::UnsupportedOperation { operation: "predict_probabilities", .. }`. A decision
function SHALL expose the raw linear scores.

#### Scenario: Probabilities are available for log_loss

- **WHEN** a model fitted with `log_loss` predicts probabilities
- **THEN** each row lies in `[0, 1]` and sums to one

#### Scenario: Probabilities are refused for hinge

- **WHEN** a model fitted with `hinge` is asked for probabilities
- **THEN** the call returns an `UnsupportedOperation` error

#### Scenario: Labels are the arg-max of the decision function

- **WHEN** predictions and decisions are requested for the same features
- **THEN** each predicted label is the index of the largest decision value

### Requirement: Each classifier loss has its documented behavior

The estimator SHALL support the five classifier losses. `hinge` and `squared_hinge` SHALL be
zero beyond a unit margin; `perceptron` SHALL update only on a misclassified sample;
`log_loss` SHALL be the logistic negative log-likelihood; `modified_huber` SHALL be smooth
inside the margin band and linear beyond it. A linearly separable design SHALL be classified
with zero training error by `hinge` and `perceptron` after enough epochs.

#### Scenario: A separable design is learned exactly

- **WHEN** a linearly separable design is fitted with `hinge` and enough epochs
- **THEN** the training labels are all predicted correctly

#### Scenario: Huber correction is linear beyond the band

- **WHEN** the same design is fitted with `modified_huber`
- **THEN** the fit converges and probabilities are available

### Requirement: Class weighting reweights the data term

The classifier SHALL accept the shared class-weighting policy (`None`, `Balanced`,
`Explicit`) and apply it through the substrate, expanding it into per-sample weights without
penalizing the intercept. `Balanced` SHALL improve recall of the minority class on an
imbalanced design.

#### Scenario: Balanced improves the minority class

- **WHEN** an imbalanced design is fitted with and without `Balanced`
- **THEN** the balanced fit predicts the minority class at least as often on the training data

#### Scenario: Explicit weights follow the class table

- **WHEN** explicit per-class weights are supplied
- **THEN** they are applied in the canonical class order

### Requirement: Warm start uses an explicit coefficient source

`warm_start_from` SHALL accept coefficients and intercepts shaped for the current class
table and SHALL seed the learning loop from them; a missing source SHALL start from zero.
Supplying a source whose class table or feature count disagrees SHALL return a structured
error. The estimator SHALL stay immutable and `Send + Sync`.

#### Scenario: Seeded refit improves over a cold start

- **WHEN** a model's coefficients are used as the warm start for a refit on the same data
- **THEN** the refit starts from them and, within the same epoch budget, is no worse than the
  cold start

#### Scenario: Misaligned warm start is rejected

- **WHEN** the warm-start source carries a different number of features
- **THEN** the build or fit returns a structured error

### Requirement: Dense, sparse, streaming and incremental learning share one model

`fit` SHALL accept dense and CSR designs; `fit_streaming` SHALL consume a supervised one-shot
source in a single pass without retaining the design; `partial_fit` on the fitted model SHALL
continue the schedule and the average. A label unseen at fit time SHALL be rejected by
`partial_fit`. The streaming and incremental fits SHALL agree with the equivalent in-memory
epochs within the documented tolerance under the same seed.

#### Scenario: One streaming pass equals one in-memory epoch

- **WHEN** the same data is fitted in one in-memory epoch and streamed once in the same order
  without shuffling
- **THEN** the coefficients agree within tolerance

#### Scenario: Incremental continuation improves the fit

- **WHEN** `partial_fit` is called with more correctly classified data after `fit`
- **THEN** the training objective does not increase

#### Scenario: An unseen class is rejected

- **WHEN** `partial_fit` receives a label the fit never saw
- **THEN** it returns a structured error

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
