# Spec Delta

## Purpose

Defines `SKIsolationForest`: random isolation trees, path-length scoring, the contamination
offset, representation parity, determinism and validation.

## ADDED Requirements

### Requirement: Builder-only construction with documented defaults

`SKIsolationForest` SHALL be constructible only through its builder, exposing
`number_of_trees` (default 100), `maximum_samples` (`Auto` default, meaning
`min(256, n_samples)`), `contamination` (`Auto` default or a fraction in `(0, 0.5]`),
`maximum_features` (default 1.0), `bootstrap` (default false), `random_seed` (default 0),
`backend` (default `Faer`) and `execution_mode` (default `Automatic`).

#### Scenario: Default configuration

- **WHEN** a detector is built without calling any setter
- **THEN** it uses 100 trees, at most 256 samples per tree, automatic contamination, all
  features, no bootstrap, `Faer` and `Automatic`

### Requirement: Isolation tree growth

Each tree SHALL be grown on a seeded subsample by repeatedly choosing a random feature and
a random split threshold between that feature's observed minimum and maximum, until a node
has one sample or reaches the isolation depth limit.

#### Scenario: A single point is isolated

- **WHEN** a node contains one sample
- **THEN** it becomes a leaf with depth-plus-size path length

#### Scenario: Split is within the observed range

- **WHEN** a node splits on a feature
- **THEN** the threshold lies between the node's minimum and maximum for that feature

### Requirement: Path-length scoring

The detector SHALL compute each row's expected path length across trees and convert it to
the anomaly score `2^(-E[h] / c(n))`, where `c(n)` is the average path length of an
unsuccessful binary-search tree with `n` samples.

#### Scenario: Shallow isolation scores higher

- **WHEN** an anomalous row is isolated at a shallower depth than a normal row
- **THEN** its anomaly score is larger

#### Scenario: Score is bounded

- **WHEN** any row is scored
- **THEN** the anomaly score is in `[0, 1]` and never NaN

### Requirement: Contamination offset

The detector SHALL derive the decision offset from `contamination`: an explicit fraction
uses that quantile of the training scores, and `Auto` uses the documented fixed offset.
The offset SHALL be exposed on the model.

#### Scenario: Explicit contamination selects a quantile

- **WHEN** `contamination` is a fraction
- **THEN** the offset is the corresponding quantile of the training score samples

#### Scenario: Auto uses the fixed offset

- **WHEN** `contamination` is `Auto`
- **THEN** the documented fixed offset is used and exposed

### Requirement: Label, decision and score outputs

The model SHALL expose `predict` (`-1`/`1`), `decision_function` and `score_samples` for
dense and sparse inputs, and SHALL implement `SKOutlierPredictor<F>`.

#### Scenario: Outliers are labeled

- **WHEN** a row's decision value is negative
- **THEN** `predict` returns `-1`

#### Scenario: Batch shapes

- **WHEN** `m` rows are scored
- **THEN** labels have length `m` and scores have length `m`

### Requirement: Representation parity

Fit and prediction SHALL accept dense and sparse views, dispatch once and never densify a
sparse input.

#### Scenario: Sparse fit equals dense fit

- **WHEN** the same logical data is fitted densely and as CSR with the same seed
- **THEN** scores and labels agree within the documented tolerance

### Requirement: Validation

The detector SHALL reject empty input, zero trees, invalid sample size, contamination
outside `(0, 0.5]`, feature-count mismatch and non-finite values.

#### Scenario: Invalid contamination is rejected

- **WHEN** `contamination` is zero, negative or greater than `0.5`
- **THEN** construction returns `InvalidHyperparameter`

#### Scenario: Empty input is rejected

- **WHEN** fit receives zero rows
- **THEN** it returns `EmptyInput`

### Requirement: Determinism and concurrency

A fixed seed and input SHALL produce identical models and scores across repeated fits and
thread counts, and a fitted model SHALL be safely shareable.

#### Scenario: Repeated fits agree

- **WHEN** the same detector fits the same data twice
- **THEN** scores, labels and offsets are identical

#### Scenario: Concurrent prediction agrees

- **WHEN** several threads predict with the same fitted model
- **THEN** every result equals the sequential result
