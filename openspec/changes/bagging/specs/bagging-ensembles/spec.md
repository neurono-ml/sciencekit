# Spec Delta

## Purpose

Defines the generic bagging classifier and regressor: the base-estimator contract,
bootstrap row and feature sampling, feature-subset routing, prediction aggregation,
out-of-bag estimation, determinism and validation.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

Both estimators SHALL be constructible only through their builders, exposing
`number_of_estimators` (default 10), `maximum_samples` (default all), `maximum_features`
(default all), `bootstrap` (default true), `bootstrap_features` (default false),
`oob_score` (default false), `random_seed` (default 0), `backend` (default `Faer`) and
`execution_mode` (default `Automatic`).

#### Scenario: Default configuration

- **WHEN** a bagging estimator is built without calling any setter
- **THEN** it uses 10 base estimators, all samples and features, bootstrap rows, no feature
  bootstrap, no OOB score, `Faer` and `Automatic`

### Requirement: Generic base-estimator contract

`SKBaggingClassifier<B>` and `SKBaggingRegressor<B>` SHALL be generic over the configured
base estimator `B` and SHALL require only the existing fit and prediction contracts: `B`
fits the matching supervision axis and its fitted model implements the classifier or
regressor predictor. No new monolithic `SKEstimator` trait is required.

#### Scenario: A tree base estimator fits

- **WHEN** `B` is a decision-tree classifier configuration
- **THEN** bagging fits it on every bootstrap sample and aggregates its probabilities

#### Scenario: A non-tree base estimator fits

- **WHEN** `B` is a linear or neighbour classifier configuration
- **THEN** bagging composes it without changes to the base estimator

#### Scenario: Wrong supervision axis does not compile

- **WHEN** a regressor configuration is used with the bagging classifier
- **THEN** the code does not compile because the base fit/predict bounds are not satisfied

### Requirement: Bootstrap rows and features

Fit SHALL draw a bootstrap row sample per estimator, and optionally a feature subset when
`bootstrap_features` is enabled, using deterministic per-estimator seeds derived from
`random_seed`.

#### Scenario: Bootstrap rows are reproducible

- **WHEN** the same bagging estimator fits the same data twice
- **THEN** every estimator receives the same rows

#### Scenario: Feature bootstrap is reproduced

- **WHEN** `bootstrap_features` is enabled
- **THEN** each estimator receives the same feature subset on every fit

### Requirement: Feature-subset routing

When an estimator is fitted on a feature subset, the bagging model SHALL route prediction
inputs through that same subset before calling the estimator and SHALL not leak the full
feature matrix to it.

#### Scenario: Prediction uses the estimator's subset

- **WHEN** an estimator was fitted on features `[0, 2]`
- **THEN** prediction for that estimator reads only columns `0` and `2`

#### Scenario: Subsets are stored on the model

- **WHEN** feature bootstrap is enabled
- **THEN** the model exposes the feature subset of every estimator

### Requirement: Prediction aggregation

The classifier SHALL average the base estimators' class probabilities and predict the
largest, breaking ties by the smallest canonical class index; the regressor SHALL average
the base responses.

#### Scenario: Classifier probabilities are averages

- **WHEN** two base estimators return class-0 probabilities `0.2` and `0.6`
- **THEN** the bagging probability for class `0` is `0.4`

#### Scenario: Regressor response is the mean

- **WHEN** three base estimators predict `1.0`, `2.0` and `6.0`
- **THEN** the bagging prediction is `3.0`

### Requirement: Out-of-bag estimation

When `oob_score` is enabled, the model SHALL compute the OOB score from the rows absent
from each estimator's bootstrap sample — accuracy for classification, R² for regression.

#### Scenario: OOB score uses only out-of-bag rows

- **WHEN** OOB scoring is enabled
- **THEN** the score uses only predictions from estimators that did not see the row

#### Scenario: OOB without bootstrap is rejected

- **WHEN** `oob_score` is enabled while `bootstrap` is false
- **THEN** construction returns `InvalidHyperparameter`

### Requirement: Parallel and deterministic fitting

Base estimators SHALL be fitted in parallel with each writing its own slot, and the
resulting model SHALL be identical across repeated fits and thread counts.

#### Scenario: Parallel fit equals sequential fit

- **WHEN** the same estimator is fitted sequentially and through the parallel path
- **THEN** both models are identical

#### Scenario: Thread count does not change the model

- **WHEN** fit runs with different rayon thread counts
- **THEN** predictions and feature subsets are identical

### Requirement: Representation behavior

Bagging SHALL pass the input representation through to the base estimator unchanged; a
representation the base estimator rejects SHALL surface that estimator's structured error,
and bagging SHALL not densify it on its own.

#### Scenario: Sparse-capable base estimator receives sparse input

- **WHEN** `B` supports CSR and the training view is sparse
- **THEN** the base estimator receives the sparse view and no dense copy is made

#### Scenario: Dense-only base estimator reports the rejection

- **WHEN** `B` is dense-only and the training view is sparse
- **THEN** the base estimator's `UnsupportedRepresentation` error reaches the caller

### Requirement: Validation

Bagging SHALL reject empty input, a zero estimator count, invalid sample/feature
fractions, feature-count mismatch and non-finite values with structured central errors.

#### Scenario: Zero estimator count is rejected

- **WHEN** `number_of_estimators` is zero
- **THEN** construction returns `InvalidHyperparameter` naming the field

#### Scenario: Invalid fraction is rejected

- **WHEN** a sample or feature fraction is outside `(0, 1]`
- **THEN** construction returns `InvalidHyperparameter`

#### Scenario: Empty input is rejected

- **WHEN** fit receives zero rows
- **THEN** it returns `EmptyInput` naming the operation

### Requirement: Concurrent prediction

Fitted bagging models SHALL be safely shareable across threads and SHALL produce the same
result under concurrent prediction as under a single thread.

#### Scenario: One model serves several threads

- **WHEN** several threads predict with the same bagging model
- **THEN** every call completes without locking and returns its sequential result
