# gradient-boosting Specification

## Purpose
Defines the gradient-boosting classifier and regressor: the loss/gradient/Hessian
contract, stage-wise additive fitting with Newton leaf values, shrinkage, stochastic
subsampling, early stopping, prediction and validation.

## Requirements

### Requirement: Builder-only construction with documented defaults

Both estimators SHALL be constructible only through their builders, exposing the
documented stage, learning-rate, loss, tree-constraint, subsampling, regularization,
early-stopping, backend and execution-mode setters.

#### Scenario: Default configuration

- **WHEN** a booster is built without calling any setter
- **THEN** it uses 100 stages, learning rate 0.1, depth 3, no subsampling, no
  regularization, no early stopping, `Faer` and `Automatic`

#### Scenario: Backend reaches the fitted model

- **WHEN** a non-default backend is configured and the booster is fitted
- **THEN** the fitted model records that backend in its prediction spans

### Requirement: Loss, gradient and Hessian contract

Each loss SHALL define the initial prediction, the negative gradient and the Hessian per
training row, and the stage fitting SHALL consume them without loss-specific branching in
the tree code.

#### Scenario: Squared-error gradient

- **WHEN** the regression loss is `SquaredError` and the current prediction is `p` for
  target `y`
- **THEN** the negative gradient is `y − p` and the Hessian is `1`

#### Scenario: Log-loss gradient

- **WHEN** the classification loss is `LogLoss` with probability `p`
- **THEN** the negative gradient is `y − p` and the Hessian is `p(1 − p)`

#### Scenario: Multiclass gradients

- **WHEN** a `c`-class problem is fitted
- **THEN** each stage produces `c` gradient/Hessian vectors and `c` trees

### Requirement: Newton leaf values and shrinkage

After a stage tree is grown on the gradients, every leaf value SHALL be replaced by
`−Σg / (Σh + λ)` and scaled by the learning rate before it is added to the model.

#### Scenario: One-leaf Newton step

- **WHEN** a stage has a single leaf with gradients `g = [2, −1]`, Hessians `h = [1, 1]`
  and `λ = 0`
- **THEN** the leaf value before shrinkage is `−0.5`

#### Scenario: Learning rate scales the update

- **WHEN** the learning rate is `0.1`
- **THEN** every leaf contribution is multiplied by `0.1` before being added

### Requirement: Stage-wise additive prediction

The model SHALL predict as the initial prediction plus the sum of the stage trees,
scaled by the learning rate, and the classifier SHALL expose labels and probabilities
through the configured link.

#### Scenario: Regression sums the stages

- **WHEN** a regressor has initial prediction `2.0` and one stage tree predicting `0.5`
  with learning rate `0.1`
- **THEN** the prediction is `2.05`

#### Scenario: Binary probabilities are sigmoid of the score

- **WHEN** a binary classifier scores a row at `0.0`
- **THEN** the probability of the positive class is `0.5`

#### Scenario: Multiclass probabilities sum to one

- **WHEN** a multiclass classifier scores a row
- **THEN** the softmax probabilities sum to one

### Requirement: Stochastic subsampling

When `subsample < 1`, each stage SHALL fit its tree on a seeded row subset, and when
`maximum_features` is restricted each split SHALL consider only the selected features.

#### Scenario: Subsample is reproducible

- **WHEN** the same `random_seed` and data are fitted twice
- **THEN** the same rows and features are selected at every stage

#### Scenario: Full subsample uses every row

- **WHEN** `subsample = 1.0`
- **THEN** every stage tree sees every training row

### Requirement: Early stopping

When `n_iter_no_change` is set, the booster SHALL hold out `validation_fraction` of the
rows, monitor the validation loss and stop after the configured number of non-improving
stages; the model SHALL report the number of stages actually fitted.

#### Scenario: Early stop reports fewer stages

- **WHEN** the validation loss stops improving for `n_iter_no_change` stages
- **THEN** fitting stops early and the model reports the reduced stage count

#### Scenario: Disabled early stopping fits every stage

- **WHEN** `n_iter_no_change` is not set
- **THEN** the model fits exactly `number_of_estimators` stages

### Requirement: Feature importances

The model SHALL expose normalized feature importances aggregated from the stage trees and
summing to one when at least one split exists.

#### Scenario: Importances aggregate the stages

- **WHEN** two stages assign different importances to a feature
- **THEN** the booster importance is their sample-weighted aggregate

### Requirement: Representation parity

Both boosters SHALL accept dense and borrowed CSR features for fit and prediction and
SHALL never densify a sparse input.

#### Scenario: Sparse fit equals dense fit

- **WHEN** the same logical design is fitted densely and as CSR with the same seed
- **THEN** predictions agree within the documented tolerance

#### Scenario: Sparse prediction is supported

- **WHEN** a booster predicts on a sparse batch
- **THEN** prediction succeeds without densifying

### Requirement: Validation

The booster SHALL reject empty input, non-positive learning rate or estimator count,
invalid subsample/feature/regularization parameters, an invalid validation fraction,
single-class classification, non-finite values and feature-count mismatch.

#### Scenario: Non-positive learning rate is rejected

- **WHEN** `learning_rate` is zero or negative
- **THEN** construction returns `InvalidHyperparameter`

#### Scenario: Invalid subsample is rejected

- **WHEN** `subsample` is outside `(0, 1]`
- **THEN** construction returns `InvalidHyperparameter`

#### Scenario: Single-class classification is rejected

- **WHEN** the classifier receives only one class
- **THEN** fit returns `InvalidHyperparameter` naming the class requirement

### Requirement: Determinism and concurrency

A fixed configuration and input SHALL produce identical models and predictions across
repeated fits and thread counts, and a fitted model SHALL be safely shareable.

#### Scenario: Repeated fits agree

- **WHEN** the same booster fits the same data twice
- **THEN** stage trees, leaf values and predictions are identical

#### Scenario: Concurrent prediction agrees

- **WHEN** several threads predict with the same fitted model
- **THEN** every result equals the sequential result
