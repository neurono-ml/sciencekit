# Spec Delta

## Purpose

Defines histogram gradient boosting: binned feature representation, histogram-based split
finding, leaf-wise growth, categorical partitions, missing-value routing, early stopping
and the classifier/regressor prediction surface.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

Both estimators SHALL be constructible only through their builders, exposing the
documented iteration, binning, leaf, regularization, feature, categorical,
early-stopping, backend and execution-mode setters.

#### Scenario: Default configuration

- **WHEN** a histogram booster is built without calling any setter
- **THEN** it uses 100 iterations, 255 bins, 31 leaves, at least 20 samples per leaf, no
  categorical features, early stopping with 10 non-improving iterations, `Faer` and
  `Automatic`

### Requirement: Quantile binning with a missing bin

Fit SHALL derive at most `max_bins` quantile bin edges per numeric feature from the
training data, assign every value to a bin, and reserve a dedicated bin for missing
values.

#### Scenario: Bins respect the maximum

- **WHEN** a numeric feature has more distinct values than `max_bins`
- **THEN** the stored edges produce at most `max_bins` bins

#### Scenario: Missing values get the missing bin

- **WHEN** a feature contains a missing value
- **THEN** that value is assigned to the dedicated missing bin and is not rejected

#### Scenario: Binning is deterministic

- **WHEN** the same data is fitted twice
- **THEN** the derived edges and bin assignments are identical

### Requirement: Histogram split finding

For every node the booster SHALL accumulate gradient and Hessian sums per feature/bin,
evaluate the split gain for every valid bin boundary, and select the largest gain subject
to the leaf constraints; feature histograms SHALL be computed in parallel.

#### Scenario: Histogram gain matches a hand-computed split

- **WHEN** a node has known gradients and Hessians in two bins
- **THEN** the selected split is the one with the larger computed gain

#### Scenario: Constraint blocks a split

- **WHEN** a candidate split would leave fewer than `minimum_samples_leaf` samples on a
  side
- **THEN** that split is rejected

#### Scenario: Leaf-wise growth respects the node budget

- **WHEN** `max_leaf_nodes` is reached
- **THEN** no further split is made and the remaining nodes become leaves

### Requirement: Histogram subtraction

When a node is split, the child histograms SHALL be derived from the parent histogram by
subtraction where it is cheaper, without changing the resulting split gains.

#### Scenario: Subtraction equals direct accumulation

- **WHEN** a child histogram is computed by subtraction and by direct accumulation
- **THEN** both histograms produce the same gain within the documented tolerance

### Requirement: Missing-value routing

Each split SHALL learn a default direction for rows in the missing bin, and prediction
SHALL route a missing value through that direction.

#### Scenario: Missing rows follow the learned direction

- **WHEN** a split learns that missing rows go left
- **THEN** a query with a missing value at that feature traverses the left child

#### Scenario: Missing routing is deterministic

- **WHEN** the same data is fitted twice
- **THEN** the learned missing directions are identical

### Requirement: Categorical partitions

Features declared categorical SHALL be split by partitioning their category bins into two
groups using the gradient statistics, represented by a `bitvec` membership mask, without
imposing an ordinal order on the categories.

#### Scenario: Categories are partitioned, not ordered

- **WHEN** a categorical feature has categories `{a, b, c}` and the optimal split separates
  `{a, c}` from `{b}`
- **THEN** the split is represented as a membership mask, not as a threshold on category
  codes

#### Scenario: Unknown categorical value at prediction

- **WHEN** prediction receives a category not seen at fit
- **THEN** the value follows the documented unknown-category direction and no panic occurs

### Requirement: Boosting and early stopping

The booster SHALL reuse the gradient/Hessian and Newton-leaf contract of gradient
boosting, apply the learning rate to every leaf, and stop early when the validation loss
does not improve for `n_iter_no_change` iterations.

#### Scenario: Early stop reports the fitted iteration count

- **WHEN** validation loss stops improving
- **THEN** fitting stops and the model reports the reduced iteration count

#### Scenario: Learning rate scales the leaf

- **WHEN** the learning rate is `0.1`
- **THEN** each leaf contribution is multiplied by `0.1`

### Requirement: Binned prediction

Prediction SHALL bin raw dense or sparse query features with the stored edges and
traverse the histogram tree; a sparse query SHALL be read without densification.

#### Scenario: Dense and sparse queries agree

- **WHEN** the same logical query is supplied densely and as CSR
- **THEN** both predictions are identical

#### Scenario: Sparse query is not densified

- **WHEN** a sparse query is predicted
- **THEN** only its stored entries are read

### Requirement: Representation parity

Both estimators SHALL accept dense and borrowed CSR features for fit and prediction and
SHALL never densify a sparse input.

#### Scenario: Sparse fit equals dense fit

- **WHEN** the same logical design is fitted densely and as CSR with the same seed
- **THEN** predictions agree within the documented tolerance

### Requirement: Validation

The booster SHALL reject empty input, invalid bin/leaf/learning parameters, unknown
categorical columns, single-class classification, non-finite values outside the missing
marker, feature-count mismatch and an invalid validation fraction.

#### Scenario: Invalid bin count is rejected

- **WHEN** `max_bins` is zero or greater than 255
- **THEN** construction returns `InvalidHyperparameter`

#### Scenario: Unknown categorical column is rejected

- **WHEN** a categorical feature index is outside the fitted column range
- **THEN** construction returns `InvalidHyperparameter` naming the column

#### Scenario: Single-class classification is rejected

- **WHEN** the classifier receives only one class
- **THEN** fit returns `InvalidHyperparameter` naming the class requirement

### Requirement: Determinism and concurrency

A fixed configuration and input SHALL produce identical models and predictions across
repeated fits and thread counts, and a fitted model SHALL be safely shareable.

#### Scenario: Repeated fits agree

- **WHEN** the same booster fits the same data twice
- **THEN** bins, splits, leaf values and predictions are identical

#### Scenario: Concurrent prediction agrees

- **WHEN** several threads predict with the same fitted model
- **THEN** every result equals the sequential result
