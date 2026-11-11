# Spec Delta

## Purpose

Defines the random-forest classifier and regressor: bootstrap sampling, per-tree feature
subsampling, parallel lock-free fitting, out-of-bag estimation, aggregation, feature
importances, representation parity, determinism and validation.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

Both estimators SHALL be constructible only through their builders, exposing
`number_of_trees` (default 100), `criterion`, `maximum_depth`, `minimum_samples_split`,
`minimum_samples_leaf`, `maximum_features` (`Sqrt` default), `bootstrap` (default true),
`maximum_samples` (default all), `random_seed` (default 0), `class_weight` (classifier),
`oob_score` (default false), `backend` (default `Faer`) and `execution_mode` (default
`Automatic`).

#### Scenario: Default configuration

- **WHEN** a forest is built without calling any setter
- **THEN** it uses 100 trees, bootstrap sampling, square-root feature subsampling, all
  samples, no OOB score, `Faer` and `Automatic`

#### Scenario: Backend reaches the fitted model

- **WHEN** a non-default backend is configured and the forest is fitted
- **THEN** the fitted model records that backend in its prediction spans

### Requirement: Deterministic per-tree bootstrap

Each tree SHALL be fitted on a bootstrap sample drawn with the seeded generator from
`random_seed` and the tree index, so the forest is identical across repeated fits and
thread counts.

#### Scenario: Repeated fits agree

- **WHEN** the same forest fits the same data twice
- **THEN** every tree structure, leaf value and prediction is identical

#### Scenario: Thread count does not change the forest

- **WHEN** the same fit runs with different rayon thread counts
- **THEN** the resulting trees and predictions are identical

### Requirement: Parallel lock-free tree fitting

Trees SHALL be fitted in parallel with each writing only its own result slot, and no shared
mutable accumulator SHALL be used during fit.

#### Scenario: Parallel fit equals sequential fit

- **WHEN** the same forest is fitted sequentially and through the parallel path
- **THEN** both forests are identical

### Requirement: Feature subsampling per split

Every split SHALL consider only the features selected by `maximum_features` for that tree
and node, using the seeded generator, with `Sqrt` as the forest default.

#### Scenario: Sqrt selects the square root of features

- **WHEN** `maximum_features(Sqrt)` is configured on a nine-feature problem
- **THEN** each split considers three features

#### Scenario: All selects every feature

- **WHEN** `maximum_features(All)` is configured
- **THEN** every split considers every feature

### Requirement: Classification aggregation

The classifier SHALL predict by averaging the per-tree class probabilities and selecting
the largest average, breaking ties by the smallest canonical class index.

#### Scenario: Probabilities are forest averages

- **WHEN** two trees return class-0 probabilities `0.2` and `0.6`
- **THEN** the forest probability for class `0` is `0.4`

#### Scenario: Ties are deterministic

- **WHEN** two classes share the largest average probability
- **THEN** the class with the smaller canonical index is predicted

### Requirement: Regression aggregation

The regressor SHALL predict the average of the per-tree responses.

#### Scenario: Mean of tree responses

- **WHEN** three trees predict `1.0`, `2.0` and `6.0`
- **THEN** the forest prediction is `3.0`

### Requirement: Out-of-bag estimation

When `oob_score` is enabled, each tree SHALL record the training rows absent from its
bootstrap sample, and the model SHALL expose an OOB score computed only on those rows —
accuracy for classification, R² for regression.

#### Scenario: OOB score uses only out-of-bag rows

- **WHEN** OOB scoring is enabled
- **THEN** the reported score is computed from predictions made by trees that did not see
  the row in their bootstrap sample

#### Scenario: OOB score is unavailable when disabled

- **WHEN** `oob_score` is false
- **THEN** the model reports no OOB score instead of fabricating one

### Requirement: Forest feature importances

The model SHALL expose normalized feature importances equal to the mean of the per-tree
importances, summing to one when at least one tree splits.

#### Scenario: Importances are averaged

- **WHEN** two trees assign different importances to a feature
- **THEN** the forest importance is their mean

### Requirement: Representation parity

Both forests SHALL accept dense and borrowed CSR features for fit and prediction, and
SHALL never densify a sparse input.

#### Scenario: Sparse fit equals dense fit

- **WHEN** the same logical design is fitted densely and as CSR with the same seed
- **THEN** the forests produce the same predictions

#### Scenario: Sparse prediction is supported

- **WHEN** a forest predicts on a sparse batch
- **THEN** prediction succeeds without densifying

### Requirement: Validation

The forest SHALL reject empty input, a zero tree count, invalid sample/depth/split/leaf
parameters, feature-count mismatch, non-finite values, single-class classification and an
OOB request without bootstrap.

#### Scenario: Zero tree count is rejected

- **WHEN** `number_of_trees` is zero
- **THEN** construction returns `InvalidHyperparameter` naming the field

#### Scenario: OOB without bootstrap is rejected

- **WHEN** `oob_score` is enabled while `bootstrap` is false
- **THEN** construction returns `InvalidHyperparameter`

#### Scenario: Single-class classification is rejected

- **WHEN** the classifier receives only one class
- **THEN** fit returns `InvalidHyperparameter` naming the class requirement

### Requirement: Concurrent prediction

Fitted forests SHALL be safely shareable across threads and SHALL produce the same result
under concurrent prediction as under a single thread.

#### Scenario: One model serves several threads

- **WHEN** several threads predict with the same forest
- **THEN** every call completes without locking and returns its sequential result
