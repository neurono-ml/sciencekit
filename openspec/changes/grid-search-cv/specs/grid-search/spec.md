# Spec Delta

## Purpose

Defines `SKGridSearchCV`: candidate evaluation across folds, score aggregation, best-model
selection, refit and the classifier/regressor prediction wrappers.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

The classifier and regressor grid searches SHALL be constructible only through their
builders, exposing a candidate list, a splitter, a scoring policy, `refit` (default true),
`error_score` (`Raise` default), an optional group vector and `execution_mode` (default
`Automatic`).

#### Scenario: Default configuration

- **WHEN** a grid search is built without optional setters
- **THEN** it refits the winner, raises on a candidate failure and uses `Automatic`
  execution

### Requirement: Candidate evaluation across folds

Fit SHALL evaluate every candidate on every fold produced by the configured splitter,
fitting on the train indices and scoring on the test indices.

#### Scenario: Every candidate sees every fold

- **WHEN** three candidates and five folds are configured
- **THEN** fifteen fits and scores are produced

#### Scenario: Folds are independent

- **WHEN** the same grid search runs with different thread counts
- **THEN** every candidate/fold score is identical within the documented tolerance

### Requirement: Score aggregation and best selection

The grid search SHALL aggregate mean and standard deviation per candidate and select the
candidate with the highest mean score, breaking ties by candidate index.

#### Scenario: Highest mean wins

- **WHEN** two candidates have different mean scores
- **THEN** the higher mean is selected

#### Scenario: Ties are deterministic

- **WHEN** two candidates have the same mean
- **THEN** the earlier candidate index is selected

### Requirement: Refit

When `refit` is true, the winner SHALL be refitted on the full data; when false, the model
SHALL expose the cross-validation results but prediction SHALL return
`UnsupportedOperation`.

#### Scenario: Refit produces a best model

- **WHEN** `refit` is true and fitting completes
- **THEN** the model exposes a best fitted model trained on all rows

#### Scenario: No refit blocks prediction

- **WHEN** `refit` is false
- **THEN** prediction returns `UnsupportedOperation`

### Requirement: Classifier and regressor prediction

The classifier wrapper SHALL implement `SKClassifierPredictor` and the regressor wrapper
`SKRegressorPredictor`, delegating to the best fitted model.

#### Scenario: Classifier labels and probabilities

- **WHEN** a classifier grid search predicts
- **THEN** labels and probabilities equal the best model's outputs

#### Scenario: Regressor responses

- **WHEN** a regressor grid search predicts
- **THEN** responses equal the best model's outputs

### Requirement: Error score

A candidate failure on a fold SHALL either raise the underlying error or contribute the
configured numeric `error_score`, according to the builder configuration.

#### Scenario: Raise policy

- **WHEN** a candidate fails and `error_score = Raise`
- **THEN** fit returns the candidate's structured error

#### Scenario: Numeric policy

- **WHEN** a candidate fails and a numeric error score is configured
- **THEN** that value is recorded for the failing fold and fitting continues

### Requirement: Representation parity

Fold gathering SHALL accept dense and sparse features and SHALL never densify a sparse
input.

#### Scenario: Sparse grid search

- **WHEN** a sparse feature matrix is searched
- **THEN** every fold receives a CSR train/test block and no dense copy is made

### Requirement: Validation

The grid search SHALL reject an empty candidate list, an incompatible scorer, invalid
groups, a splitter error and prediction without refit.

#### Scenario: Empty candidates are rejected

- **WHEN** no candidate is supplied
- **THEN** construction returns `InvalidHyperparameter`

#### Scenario: Incompatible scorer is rejected

- **WHEN** a regressor scorer is paired with a classifier estimator
- **THEN** construction returns `InvalidHyperparameter`

### Requirement: Concurrency and determinism

Candidate/fold evaluation SHALL be parallel and deterministic, with results collected by
candidate and fold index rather than completion order.

#### Scenario: One model serves several threads

- **WHEN** concurrent predictions use the same fitted grid-search model
- **THEN** every result equals the sequential result
